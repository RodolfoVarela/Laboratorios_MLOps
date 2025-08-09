# src/baselines.py
import os, copy, torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_simplecnn_loaders(train_dir, val_dir, batch_size=32, num_workers=2):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
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

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 112x112
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 56x56
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 28x28
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*28*28, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def train_simplecnn(train_dir, val_dir, device, epochs=5, lr=1e-3, batch_size=32):
    loaders, sizes, classes = get_simplecnn_loaders(train_dir, val_dir, batch_size=batch_size)
    model = SimpleCNN(num_classes=len(classes)).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_w, best_acc = copy.deepcopy(model.state_dict()), 0.0

    for epoch in range(epochs):
        for phase in ["train", "val"]:
            model.train() if phase=="train" else model.eval()
            loss_sum, correct = 0.0, 0
            for x, y in loaders[phase]:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                with torch.set_grad_enabled(phase=="train"):
                    out = model(x)
                    loss = crit(out, y)
                    if phase=="train":
                        loss.backward(); opt.step()
                loss_sum += loss.item()*x.size(0)
                correct += (out.argmax(1)==y).sum().item()
            acc = correct / sizes[phase]
            if phase=="val" and acc > best_acc:
                best_acc = acc; best_w = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_w)
    # Evaluación final en val
    model.eval()
    ys, yh = [], []
    with torch.no_grad():
        for x, y in loaders["val"]:
            x = x.to(device)
            out = model(x)
            yh.extend(out.argmax(1).cpu().numpy())
            ys.extend(y.numpy())
    acc = accuracy_score(ys, yh)
    cm = confusion_matrix(ys, yh)
    rep = classification_report(ys, yh, output_dict=True, zero_division=0)

    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("outputs","simplecnn_best.pth"))
    return acc, cm, rep, classes

