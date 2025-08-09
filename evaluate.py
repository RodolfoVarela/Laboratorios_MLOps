# src/evaluate.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def guardar_matriz_confusion(cm, class_names, out_png):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def tabla_clasification_report(report_dict, out_csv):
    """report_dict = classification_report(..., output_dict=True)"""
    df = pd.DataFrame(report_dict).T
    df.to_csv(out_csv, index=True)
    return df

# --- Mosaico de ejemplos por clase ---
import random
from PIL import Image
import matplotlib.pyplot as plt
import math
import os

def mosaic_examples_from_imagefolder(imagefolder, class_names, k_per_class=6, out_png=None, seed=42):
    """
    Crea un mosaico con k_per_class imágenes por clase usando paths de ImageFolder (sin transforms).
    """
    random.seed(seed)
    # 'samples' = lista de (path, class_idx)
    paths_by_class = {i: [] for i in range(len(class_names))}
    for path, cls in imagefolder.samples:
        paths_by_class[cls].append(path)

    rows = len(class_names)
    cols = k_per_class
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.2, rows*2.2))
    if rows == 1: axes = [axes]

    for r, cls_idx in enumerate(range(len(class_names))):
        pick = random.sample(paths_by_class[cls_idx], min(k_per_class, len(paths_by_class[cls_idx])))
        for c in range(cols):
            ax = axes[r][c] if cols > 1 else axes[r]
            ax.axis("off")
            if c < len(pick):
                img = Image.open(pick[c]).convert("RGB")
                ax.imshow(img)
            if c == 0:
                ax.set_title(class_names[cls_idx], fontsize=9, loc="left")
    plt.tight_layout()
    if out_png:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=150)
        plt.close(fig)
    else:
        plt.show()

# --- Reliability plot (multi-clase con max prob) ---
import numpy as np
import matplotlib.pyplot as plt

def plot_reliability_from_model(model, X_val, y_val, out_png, n_bins=10):
    """
    Usa predict_proba del modelo (p.ej., RandomForest) para construir:
    - bins de confianza (max prob)
    - accuracy por bin
    """
    proba = model.predict_proba(X_val)  # (N, C)
    y_pred = np.argmax(proba, axis=1)
    conf = np.max(proba, axis=1)        # confianza del modelo
    correct = (y_pred == y_val).astype(int)

    bins = np.linspace(0.0, 1.0, n_bins+1)
    idx = np.digitize(conf, bins) - 1
    acc_per_bin, cnt_per_bin, centers = [], [], []
    for b in range(n_bins):
        mask = (idx == b)
        cnt = mask.sum()
        cnt_per_bin.append(cnt)
        if cnt > 0:
            acc_per_bin.append(correct[mask].mean())
        else:
            acc_per_bin.append(np.nan)
        centers.append(0.5*(bins[b]+bins[b+1]))

    fig, ax1 = plt.subplots(figsize=(6,4))
    ax1.plot(centers, acc_per_bin, marker="o")
    ax1.set_ylim(0,1)
    ax1.set_xlabel("Confianza (max prob)")
    ax1.set_ylabel("Accuracy por bin")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.bar(centers, cnt_per_bin, width=(1.0/n_bins)*0.9, alpha=0.3)
    ax2.set_ylabel("Cantidad muestras")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
