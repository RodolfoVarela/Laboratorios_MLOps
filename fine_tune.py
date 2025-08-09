# src/fine_tune.py
import copy, torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_finetune_loaders(train_dir, val_dir, batch_size=32, num_workers=2):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    dsets = {
        "train": datasets.ImageFolder(train_dir, transform=train_tf),
        "val":   datasets.ImageFolder(val_dir,   transform=val_tf),
    }
    loaders = {
        "train": DataLoader(dsets["train"], batch_size=batch_size, shuffle=True,  num_workers=num_workers),
        "val":   DataLoader(dsets["val"],   batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    sizes = {k: len(dsets[k]) for k in dsets}
    classes = dsets["train"].classes
    return loaders, sizes, classes

def build_vgg19_model(num_classes, feature_extract=True):
    m = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    if feature_extract:
        for p in m.features.parameters():
            p.requires_grad = False
    in_feats = m.classifier[6].in_features
    m.classifier[6] = nn.Linear(in_feats, num_classes)
    return m

def train_finetune(model, loaders, sizes, device,
                   epochs_head=3, epochs_ft=5,
                   lr_head=1e-3, lr_ft=1e-4,
                   unfreeze_from=28):
    model = model.to(device)
    crit = nn.CrossEntropyLoss()

    def run(n_epochs, opt, sched):
        best_acc, best_w = 0.0, copy.deepcopy(model.state_dict())
        for _ in range(n_epochs):
            for phase in ["train", "val"]:
                model.train() if phase == "train" else model.eval()
                loss_sum, correct = 0.0, 0
                for x, y in loaders[phase]:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    with torch.set_grad_enabled(phase == "train"):
                        out = model(x)
                        loss = crit(out, y)
                        if phase == "train":
                            loss.backward(); opt.step()
                    loss_sum += loss.item() * x.size(0)
                    correct += (out.argmax(1) == y).sum().item()
                acc = correct / sizes[phase]
                if phase == "val" and acc > best_acc:
                    best_acc, best_w = acc, copy.deepcopy(model.state_dict())
            sched.step()
        model.load_state_dict(best_w)
        return best_acc

    # Fase 1: entrenar solo la cabeza
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_head)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.1)
    best_acc_head = run(epochs_head, opt, sch)

    # Fase 2: unfreeze último bloque conv (aprox. conv5_x)
    for i, layer in enumerate(model.features):
        if i >= unfreeze_from:
            for p in layer.parameters(): p.requires_grad = True
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_ft)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.1)
    best_acc_ft = run(epochs_ft, opt, sch)

    return model, max(best_acc_head, best_acc_ft)

def evaluate(model, loader_val, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader_val:
            x = x.to(device)
            out = model(x)
            y_pred.extend(out.argmax(1).cpu().numpy())
            y_true.extend(y.numpy())
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return acc, cm, rep
