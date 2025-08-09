# main.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch

from src.baselines import train_simplecnn
from src.feature_extractor import FeatureExtractor
from src.data_loader import get_dataloaders
from src.utils import train_val_split, gather_features, save_npz, load_npz
from src.train_classic import entrenar_modelos, resumen_resultados, svm_pca_grid, fila_resumen
from src.fine_tune import get_finetune_loaders, build_vgg19_model, train_finetune, evaluate
from src.evaluate import guardar_matriz_confusion, tabla_clasification_report
from torchvision import datasets
from src.evaluate import mosaic_examples_from_imagefolder


def main():
    print("=== Inicio ===")
    print("WD:", os.getcwd())

    original_images = r"C:\Users\fitov\Desktop\Maestría Ciencia de Datos\ML Ops\Lab2\archive\images"
    prepared = os.path.join("data", "fruits_prepared")
    train_folder = os.path.join(prepared, "train")
    val_folder   = os.path.join(prepared, "val")

    if not os.path.exists(train_folder) or not os.path.exists(val_folder):
        print("Creando split train/val en:", prepared)
        train_val_split(original_images, prepared, val_ratio=0.2)

    # loaders
    train_loader, class_names = get_dataloaders(train_folder, batch_size=32, shuffle=False)
    val_loader,   _           = get_dataloaders(val_folder,   batch_size=32, shuffle=False)
    print("Clases:", class_names)
    
    train_ds_raw = datasets.ImageFolder(train_folder)  # sin transform
    mosaic_examples_from_imagefolder(
    train_ds_raw, class_names, k_per_class=6,
    out_png=os.path.join("outputs", "mosaic_train.png")
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    extractor = FeatureExtractor(device=device)

    # IMPORTANTE: entrenar con stages más profundos es más liviano y suele rendir mejor.
    # stage=3 -> 512x7x7 = 25.088 features
    stages = [1, 2, 3]   # evita el 0 al inicio porque es gigante

    resumen_total = []
    for st in stages:
        print(f"\n--- Stage {st} ---")
        Xtr, ytr = gather_features(train_loader, extractor, stage=st)
        Xva, yva = gather_features(val_loader, extractor, stage=st)
        save_npz(os.path.join("features", f"stage{st}.npz"), Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva)

        resultados = entrenar_modelos(Xtr, ytr, Xva, yva)
        df = resumen_resultados(resultados, label_stage=f"stage{st}")
        print(df)
        # === SVM + PCA (grid chico) ===
        svmres = svm_pca_grid(Xtr, ytr, Xva, yva, n_comp=192)
        df_svm = fila_resumen("SVM-PCA-grid", f"stage{st}", svmres["accuracy"])
        print(df_svm)

        # Guardar tabla combinada (clásicos + SVM-PCA)
        df_total = pd.concat([df, df_svm], ignore_index=True)
        df_total.to_csv(os.path.join("outputs", f"accuracy_stage{st}.csv"), index=False)

        # === Reliability del RF de ESTE stage ===
        from src.evaluate import plot_reliability_from_model  # (puede estar arriba del archivo también)
        rf_model = resultados["RF-200"]["model"]
        plot_reliability_from_model(
            rf_model, Xva, yva,
            out_png=os.path.join("outputs", f"reliability_stage{st}_RF-200.png"),
            n_bins=10
            )

        # (opcional) guardar confusion/report del SVM-PCA
        guardar_matriz_confusion(
            svmres["confusion_matrix"], class_names,
            os.path.join("outputs", f"cm_stage{st}_SVM-PCA-grid.png")
            )
        tabla_clasification_report(
        svmres["report"],
        os.path.join("outputs", f"report_stage{st}_SVM-PCA-grid.csv")
        )

        # guardar confusions y reports
        for nombre, res in resultados.items():
            cm = res["confusion_matrix"]
            out_png = os.path.join("outputs", f"cm_stage{st}_{nombre}.png")
            guardar_matriz_confusion(cm, class_names, out_png)
            tabla_clasification_report(res["report"], os.path.join("outputs", f"report_stage{st}_{nombre}.csv"))

        resumen_total.append(df_total)

    # --- Concat Stage 2 + 3 ---
    path2 = os.path.join("features", "stage2.npz")
    path3 = os.path.join("features", "stage3.npz")
    if os.path.exists(path2) and os.path.exists(path3):
        Xtr2, ytr2, Xva2, yva2 = load_npz(path2)
    Xtr3, ytr3, Xva3, yva3 = load_npz(path3)
    assert (ytr2 == ytr3).all() and (yva2 == yva3).all(), "Las etiquetas no coinciden entre stage2 y stage3"

    Xtr = np.hstack([Xtr2, Xtr3])
    Xva = np.hstack([Xva2, Xva3])
    ytr, yva = ytr3, yva3

    print("\n--- Stage 2+3 (concat) ---")
    res_concat = entrenar_modelos(Xtr, ytr, Xva, yva)
    df_concat = resumen_resultados(res_concat, label_stage="stage2+3")

    # SVM + PCA sobre concat (usa versión robusta; si querés más veloz baja n_comp)
    svmres = svm_pca_grid(Xtr, ytr, Xva, yva, n_comp=256)
    df_svmc = fila_resumen("SVM-PCA-grid", "stage2+3", svmres["accuracy"])
    print(df_concat); print(df_svmc)

    df_concat_total = pd.concat([df_concat, df_svmc], ignore_index=True)
    df_concat_total.to_csv(os.path.join("outputs", "accuracy_stage23.csv"), index=False)

    guardar_matriz_confusion(svmres["confusion_matrix"], class_names,
                             os.path.join("outputs", "cm_stage23_SVM-PCA-grid.png"))
    tabla_clasification_report(svmres["report"],
                               os.path.join("outputs", "report_stage23_SVM-PCA-grid.csv"))
    
    # === Reliability del RF en CONCAT ===
    rf_concat = res_concat["RF-200"]["model"]
    plot_reliability_from_model(
    rf_concat, Xva, yva,   # ojo: estos Xva/yva son los de CONCATEANDO
    out_png=os.path.join("outputs", "reliability_stage23_RF-200.png"),
    n_bins=10
    )
        
    resumen_total.append(df_concat_total)
    
    # --- Fine-tuning VGG19 ---
    print("\n--- Fine-tuning VGG19 ---")
    loaders_ft, sizes_ft, classes_ft = get_finetune_loaders(train_folder, val_folder, batch_size=32)
    assert classes_ft == class_names, "Las clases difieren entre loaders!"

    model_ft = build_vgg19_model(num_classes=len(class_names), feature_extract=True)
    model_ft, _ = train_finetune(
    model_ft, loaders_ft, sizes_ft, device,
    epochs_head=3, epochs_ft=5,  # subí si querés más accuracy
    lr_head=1e-3, lr_ft=1e-4,    # ft con LR más bajo
    unfreeze_from=28             # último bloque conv
    )

    # Evaluación final FT
    acc_ft, cm_ft, rep_ft = evaluate(model_ft, loaders_ft["val"], device)
    print(f"Accuracy fine-tune: {acc_ft:.3f}")

    # Guardar outputs FT
    torch.save(model_ft.state_dict(), os.path.join("outputs", "vgg19_finetune_best.pth"))
    guardar_matriz_confusion(cm_ft, class_names, os.path.join("outputs", "cm_vgg19_finetune.png"))
    tabla_clasification_report(rep_ft, os.path.join("outputs", "report_vgg19_finetune.csv"))

    # Añadir al resumen
    df_ft = pd.DataFrame([{"stage": "fine-tune", "modelo": "VGG19-finetune", "accuracy": acc_ft}])
    resumen_total.append(df_ft)
    
    print("\n--- Baseline: SimpleCNN ---")
    acc_cnn, cm_cnn, rep_cnn, classes_cnn = train_simplecnn(
    train_folder, val_folder, device, epochs=5, lr=1e-3, batch_size=32
    )
    print(f"SimpleCNN val acc: {acc_cnn:.3f}")
    guardar_matriz_confusion(cm_cnn, classes_cnn, os.path.join("outputs","cm_simplecnn.png"))
    tabla_clasification_report(rep_cnn, os.path.join("outputs","report_simplecnn.csv"))

    # sumar al resumen final
    df_cnn = pd.DataFrame([{"stage": "baseline", "modelo": "SimpleCNN", "accuracy": acc_cnn}])
    resumen_total.append(df_cnn)

    if resumen_total:
        final = pd.concat(resumen_total, ignore_index=True)
        final.to_csv(os.path.join("outputs", "resumen_general.csv"), index=False)
        print("\nResumen general guardado en outputs/resumen_general.csv")

    print("=== Fin ===")

if __name__ == "__main__":
    main()

