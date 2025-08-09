# src/utils.py
import os, shutil, random
import numpy as np
import torch

def train_val_split(original_dir, target_root, val_ratio=0.2, seed=42):
    random.seed(seed)
    classes = [d for d in os.listdir(original_dir) if os.path.isdir(os.path.join(original_dir, d))]
    train_target = os.path.join(target_root, "train")
    val_target = os.path.join(target_root, "val")
    os.makedirs(train_target, exist_ok=True)
    os.makedirs(val_target, exist_ok=True)

    for cls in classes:
        src_folder = os.path.join(original_dir, cls)
        images = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
        random.shuffle(images)
        cut = int((1 - val_ratio) * len(images))
        split = {"train": images[:cut], "val": images[cut:]}
        for subset, imgs in split.items():
            dest_dir = train_target if subset == "train" else val_target
            class_dest = os.path.join(dest_dir, cls)
            os.makedirs(class_dest, exist_ok=True)
            for img in imgs:
                sp = os.path.join(src_folder, img)
                dp = os.path.join(class_dest, img)
                if not os.path.exists(dp):
                    shutil.copy2(sp, dp)

@torch.no_grad()
def gather_features(loader, extractor, stage=0):
    """Devuelve X (np.ndarray) y y para TODO el loader."""
    X_list, y_list = [], []
    for imgs, labels in loader:
        imgs = imgs.to(extractor.device)
        feats = extractor.extract_flattened(imgs, stage=stage).cpu().numpy()
        X_list.append(feats)
        y_list.append(labels.numpy())
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y

def save_npz(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)

def load_npz(path):
    d = np.load(path)
    return d["Xtr"], d["ytr"], d["Xva"], d["yva"]
