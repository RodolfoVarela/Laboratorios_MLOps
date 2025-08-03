# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 20:28:37 2025

@author: fitov
"""

#%% (a) Workaround OpenMP y Imports generales
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  # evita el conflicto de libiomp5md.dll

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd

# Parámetros globales
batch_size    = 64
learning_rate = 0.001
num_epochs    = 5
device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transform para CIFAR‑10 (32×32)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# Ruta base de tus carpetas train/ test y CSVs
data_root = r'C:\Users\fitov\Desktop\Maestría Ciencia de Datos\ML Ops\CIFAR-10'

#%%
import torch, os

print("torch versión:", torch.__version__)
print("CUDA toolk it:", torch.version.cuda)
print("torch.cuda.is_available():", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Primer GPU:", torch.cuda.get_device_name(0))

#%% (b) Custom Dataset y DataLoaders

class CIFAR10Local(Dataset):
    CIFAR10_CLASSES = [
        'airplane','automobile','bird','cat','deer',
        'dog','frog','horse','ship','truck'
    ]
    LABEL_MAP = {n:i for i,n in enumerate(CIFAR10_CLASSES)}

    def __init__(self, images_dir, labels_path, transform=None):
        self.images_dir = images_dir
        self.df         = pd.read_csv(labels_path)
        assert 'image_name' in self.df.columns and 'label' in self.df.columns,\
            f"Columnas inesperadas: {self.df.columns}"
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row['image_name'])
        img      = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.LABEL_MAP[row['label']]
        return img, label

# Carga datasets
full_train_ds = CIFAR10Local(
    os.path.join(data_root,'train'),
    os.path.join(data_root,'training_labels.csv'),
    transform
)
test_ds = CIFAR10Local(
    os.path.join(data_root,'test'),
    os.path.join(data_root,'test_labels.csv'),
    transform
)

# Split train (20%) / val (80%)
n_train = int(0.2*len(full_train_ds))
n_val   = len(full_train_ds) - n_train
train_ds, val_ds = random_split(full_train_ds, [n_train, n_val])

# DataLoaders
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size, shuffle=False)

print(f"→ Batches: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")


#%% (c) Exploratory Data Analysis optimizado

import matplotlib.pyplot as plt
from collections import Counter

# Conteos sin abrir imágenes
labels_str   = full_train_ds.df['label']
labels_num   = labels_str.map(CIFAR10Local.LABEL_MAP).tolist()
counts_full  = Counter(labels_num)
counts_train = Counter(labels_num[i] for i in train_ds.indices)

print("Total imágenes por clase (train completo):", counts_full)
print("Imágenes por clase (20% train):",       counts_train)

# Grid de un ejemplo por clase (10 aperturas)
firsts = full_train_ds.df.groupby('label')['image_name'].first()
fig, axes = plt.subplots(2,5,figsize=(12,6))
for ax, cls in zip(axes.flatten(), CIFAR10Local.CIFAR10_CLASSES):
    fname   = firsts[cls]
    img_pth = os.path.join(data_root,'train', fname)
    img     = Image.open(img_pth).convert('RGB')
    img_t   = transform(img)
    img_np  = img_t.permute(1,2,0).numpy()
    img_vis = (img_np*0.5)+0.5
    ax.imshow(img_vis, interpolation='nearest')
    idx = CIFAR10Local.LABEL_MAP[cls]
    ax.set_title(f"{idx}: {cls}")
    ax.axis('off')
plt.tight_layout()
plt.show()


#%% (d) Definición de Modelos

# SimpleCNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,  64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(64,128,kernel_size=5,padding=2)
        self.fc1   = nn.Linear(128*16*16,128)
        self.fc2   = nn.Linear(128,       10)
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# SimpleMLP
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3*32*32,512)
        self.fc2 = nn.Linear(512,     10)
    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


#%% (e) Funciones de entrenamiento y evaluación

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0,0,0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outs = model(imgs)
        loss = criterion(outs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*imgs.size(0)
        correct      += (outs.argmax(1)==labels).sum().item()
        total        += labels.size(0)
    return running_loss/total, correct/total

def eval_epoch(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0,0,0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outs  = model(imgs)
            loss  = criterion(outs, labels)
            running_loss += loss.item()*imgs.size(0)
            correct      += (outs.argmax(1)==labels).sum().item()
            total        += labels.size(0)
    return running_loss/total, correct/total


#%% (f) Entrenamiento SimpleCNN con checkpoint

# Instancia
model_cnn = SimpleCNN().to(device)
opt_cnn   = torch.optim.Adam(model_cnn.parameters(), lr=learning_rate)
crit      = nn.CrossEntropyLoss()

best_val_acc = 0.0
history_cnn  = {'train_loss':[], 'train_acc':[],
                'val_loss':[],   'val_acc':[]}

for ep in range(num_epochs):
    tl, ta = train_epoch(model_cnn, train_loader, crit, opt_cnn)
    vl, va = eval_epoch( model_cnn, val_loader,   crit)

    history_cnn['train_loss'].append(tl)
    history_cnn['train_acc'].append(ta)
    history_cnn['val_loss'].append(vl)
    history_cnn['val_acc'].append(va)

    print(f"CNN Ep{ep+1}/{num_epochs} → "
          f"Train: acc={ta:.3f}, Val: acc={va:.3f}")

    if va>best_val_acc:
        best_val_acc=va
        torch.save(model_cnn.state_dict(),'simplecnn_best.pth')
        print(f"  → Guardado checkpoint Ep{ep+1}, val_acc={va:.3f}")


#%% (g) Evaluación final en Test (CNN)

import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Carga mejor checkpoint
model_cnn.load_state_dict(torch.load('simplecnn_best.pth'))
model_cnn.to(device); model_cnn.eval()

# Métrica global
test_loss, test_acc = eval_epoch(model_cnn, test_loader, crit)
print(f"→ CNN Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

# Predicciones
all_p, all_y = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        outs = model_cnn(imgs)
        all_p.extend(outs.argmax(1).cpu().tolist())
        all_y.extend(labels.tolist())

# Matriz de confusión
cm = confusion_matrix(all_y, all_p)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=CIFAR10Local.CIFAR10_CLASSES,
            yticklabels=CIFAR10Local.CIFAR10_CLASSES,
            cmap='Blues')
plt.xlabel('Predicción'); plt.ylabel('Verdadero')
plt.title('CNN Confusion Matrix (Test)')
plt.show()

# Reporte
print(classification_report(all_y, all_p,
      target_names=CIFAR10Local.CIFAR10_CLASSES))


#%% (h) Entrenar SimpleMLP y comparar

# Instancia MLP
model_mlp = SimpleMLP().to(device)
opt_mlp   = torch.optim.Adam(model_mlp.parameters(), lr=learning_rate)
crit_mlp  = nn.CrossEntropyLoss()

history_mlp = {'train_acc':[], 'val_acc':[]}
for ep in range(num_epochs):
    _, ta = train_epoch(model_mlp, train_loader, crit_mlp, opt_mlp)
    _, va = eval_epoch( model_mlp, val_loader,   crit_mlp)
    history_mlp['train_acc'].append(ta)
    history_mlp['val_acc'].append(va)
    print(f"MLP Ep{ep+1}/{num_epochs} → Train: acc={ta:.3f}, Val: acc={va:.3f}")

# Test MLP
_, test_acc_mlp = eval_epoch(model_mlp, test_loader, crit_mlp)
print(f"→ MLP Test Acc: {test_acc_mlp:.4f}")


#%% (i) Modelos clásicos: RF, SVM, KNN, XGBoost

import numpy as np
from sklearn.ensemble  import RandomForestClassifier
from sklearn.svm       import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost           import XGBClassifier
from sklearn.metrics   import accuracy_score

def extract_features(ds):
    X = np.stack([img.view(-1).numpy() for img,_ in ds])
    y = np.array([lbl for _,lbl in ds])
    return X,y

X_tr, y_tr = extract_features(train_ds)
X_va, y_va = extract_features(val_ds)
X_te, y_te = extract_features(test_ds)

# Random Forest
rf = RandomForestClassifier(n_estimators=100,n_jobs=-1)
rf.fit(X_tr,y_tr)
print("RF → Val:", accuracy_score(y_va, rf.predict(X_va)),
      "Test:",accuracy_score(y_te, rf.predict(X_te)))

# SVM
svm = SVC(kernel='rbf')
svm.fit(X_tr,y_tr)
print("SVM → Val:", accuracy_score(y_va, svm.predict(X_va)),
      "Test:",accuracy_score(y_te, svm.predict(X_te)))

# KNN
knn = KNeighborsClassifier(5)
knn.fit(X_tr,y_tr)
print("KNN → Val:", accuracy_score(y_va, knn.predict(X_va)),
      "Test:",accuracy_score(y_te, knn.predict(X_te)))

# XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_tr,y_tr)
print("XGB → Val:", accuracy_score(y_va, xgb.predict(X_va)),
      "Test:",accuracy_score(y_te, xgb.predict(X_te)))


#%% (j) Transfer Learning con ResNet‑50 (Windows-friendly)

import time
import torch
import torch.nn as nn
from torchvision import models, transforms as T
from torch.utils.data import DataLoader

# 0) Configuración CUDA/cuDNN
print("Usando device:", device)
if device.type=='cuda':
    torch.backends.cudnn.benchmark = True

# 1) Transforms 128×128
tl_rt = T.Compose([
    T.Resize((128,128)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# 2) Datasets + Loaders con num_workers=0
train_ds_rt = CIFAR10Local(
    os.path.join(data_root,'train'),
    os.path.join(data_root,'training_labels.csv'),
    tl_rt
)
test_ds_rt = CIFAR10Local(
    os.path.join(data_root,'test'),
    os.path.join(data_root,'test_labels.csv'),
    tl_rt
)

train_loader_rt = DataLoader(
    train_ds_rt, batch_size=32, shuffle=True,
    num_workers=0, pin_memory=(device.type=='cuda')
)
test_loader_rt = DataLoader(
    test_ds_rt,  batch_size=32, shuffle=False,
    num_workers=0, pin_memory=(device.type=='cuda')
)

# 3) Warm‑up: comprobar loader y primer batch en GPU
imgs0, lbls0 = next(iter(train_loader_rt))
imgs0 = imgs0.to(device); lbls0 = lbls0.to(device)
print("Primer batch enviado a:", imgs0.device)

# 4) Cargar ResNet50 y congelar todo menos la fc
resnet50 = models.resnet50(pretrained=True)
for p in resnet50.parameters():
    p.requires_grad=False

in_feats    = resnet50.fc.in_features
resnet50.fc = nn.Linear(in_feats, 10)
resnet50    = resnet50.to(device)

crit_rt = nn.CrossEntropyLoss()
opt_rt  = torch.optim.Adam(resnet50.fc.parameters(), lr=1e-3)

# 5) Entrenamiento de 3 épocas con timing
for ep in range(3):
    t0 = time.time()

    # → train
    resnet50.train()
    running, corr, tot = 0,0,0
    for imgs, lbls in train_loader_rt:
        imgs = imgs.to(device)
        lbls = lbls.to(device)
        opt_rt.zero_grad()
        outs = resnet50(imgs)
        loss = crit_rt(outs, lbls)
        loss.backward()
        opt_rt.step()
        running += loss.item()*imgs.size(0)
        corr    += (outs.argmax(1)==lbls).sum().item()
        tot    += imgs.size(0)
    train_acc = corr/tot

    # → val
    vl, va = eval_epoch(resnet50, test_loader_rt, crit_rt)

    # sync y reporte
    if device.type=='cuda':
        torch.cuda.synchronize()
    print(f"Ep{ep+1}/3  time={time.time()-t0:.1f}s  Train Acc={train_acc:.3f}  Val Acc={va:.3f}")

# 6) Evaluación final en Test
test_loss_rt, test_acc_rt = eval_epoch(resnet50, test_loader_rt, crit_rt)
print(f"→ ResNet50 Test Acc (128×128): {test_acc_rt:.3f}")
