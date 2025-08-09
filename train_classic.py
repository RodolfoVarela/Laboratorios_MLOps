# src/train_classic.py
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def entrenar_modelos(X_train, y_train, X_val, y_val):
    """
    Devuelve dict con métricas y el modelo entrenado para:
      - SVM (RBF)
      - KNN (k=5)
      - RandomForest (200 árboles)
    Nota: usamos StandardScaler para SVM/KNN. with_mean=False evita alto uso de RAM.
    """
    modelos = {
        "SVM-RBF": make_pipeline(StandardScaler(with_mean=False), SVC(kernel="rbf", C=10, gamma="scale")),
        "KNN-5":   make_pipeline(StandardScaler(with_mean=False), KNeighborsClassifier(n_neighbors=5)),
        "RF-200":  RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    }

    resultados = {}
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        preds = modelo.predict(X_val)
        acc = accuracy_score(y_val, preds)
        resultados[nombre] = {
            "model": modelo,
            "accuracy": acc,
            "confusion_matrix": confusion_matrix(y_val, preds),
            "report": classification_report(y_val, preds, output_dict=True, zero_division=0)
        }
    return resultados

def resumen_resultados(resultados, label_stage):
    filas = [{"stage": label_stage, "modelo": k, "accuracy": v["accuracy"]} for k, v in resultados.items()]
    df = pd.DataFrame(filas).sort_values("accuracy", ascending=False)
    return df

from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

def svm_pca_grid(X_train, y_train, X_val, y_val, n_comp=512, cv=3):
    # límite seguro para n_components
    n_samples, n_features = X_train.shape
    n_train_fold = (n_samples * (cv - 1)) // cv  # tamaño aprox. del fold de train
    max_comp_allowed = min(n_features, n_train_fold) - 1  # ¡menos 1 para evitar el borde!
    n_comp_safe = max(2, min(n_comp, max_comp_allowed))

    pipe = make_pipeline(
        StandardScaler(with_mean=False),
        PCA(n_components=n_comp_safe, random_state=42),
        SVC(kernel="rbf", class_weight="balanced")
    )
    param_grid = {
        "svc__C": [1, 5, 10],
        "svc__gamma": ["scale", 1e-3, 1e-4],
    }
    gs = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=-1, verbose=0, error_score="raise")
    gs.fit(X_train, y_train)
    preds = gs.predict(X_val)
    acc = accuracy_score(y_val, preds)
    return {
        "model": gs.best_estimator_,
        "accuracy": acc,
        "confusion_matrix": confusion_matrix(y_val, preds),
        "report": classification_report(y_val, preds, output_dict=True, zero_division=0),
        "best_params": gs.best_params_,
    }


def fila_resumen(nombre_modelo, stage, acc):
    import pandas as pd
    return pd.DataFrame([{"stage": stage, "modelo": nombre_modelo, "accuracy": acc}])