"""
utils.py — Herramientas auxiliares para evaluación y sistema

Contiene funciones para:
  1. Sistema: Fijar semillas, crear directorios, guardar/cargar modelos.
  2. Evaluación: Calcular métricas clínicas
  3. Visualización: Graficar matrices de confusión y curvas ROC.
"""

import os
import json
import random
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
)

try:
    from src import config
except ImportError:
    print('[WARN] No se encontro config.py')
    config = None

# Utilidades 
def set_seed(seed=None):
    """
    Fija la semilla aleatoria para garantizar la reproducibilidad.
    Si no se pasa una semilla, intenta usar RANDOM_STATE de config.py.
    """
    if seed is None:
        seed = config.RANDOM_STATE if config else 42
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    print(f'Semilla global fijada en: {seed}')

def ensure_dir(directory_path):
    """
    Verifica si un directorio existe sino crea uno.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f'Carpeta creada: {directory_path}')
    return directory_path

def save_model(model, filename, save_dir=None):
    """
    Guarda un modelo entrenado en disco.
    """
    if save_dir is None:
        save_dir = config.MODELS_DIR if config else 'results/models'
    
    ensure_dir(save_dir)

    if not filename.endswith('.pkl'):
        filename += '.pkl'

    file_path = os.path.join(save_dir, filename)
    joblib.dump(model, file_path)

    print(f'Modelo guardado en: {file_path}')
    return file_path

def load_model(file_path):
    """
    Carga un modelo entrenado desde el disco.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'No se encontro el modelo en: {file_path}')
    
    model = joblib.load(file_path)
    print(f'Modelo cargado desde: {file_path}')
    return model


# Evaluacion y metricas
def evaluate_model(y_true, y_pred, y_prob=None, model_name='Modelo', experiment_name='Exp', save_dir=None):
    """
    Calcula métricas clínicas para un modelo de clasificación binaria (MDD vs HC)
    y guarda los resultados en un archivo JSON.
    
    Args:
        y_true: Array de etiquetas reales (1 = MDD, 0 = HC).
        y_pred: Array de predicciones binarias del modelo.
        y_prob: (Opcional) Array de probabilidades continuas para calcular AUC-ROC.
        model_name: Nombre del modelo (ej. 'SVM', 'RandomForest').
        experiment_name: Nombre del experimento (ej. 'FC', 'ALFF', 'Combined').
        save_dir: Directorio donde guardar el JSON. Si es None, usa config.py.
        
    Returns:
        metrics: Diccionario con todas las métricas calculadas.
    """

    # Matriz de confusion
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Metricas
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # Especificidad
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0 # Precision 
    f1 = f1_score(y_true, y_pred)

    # AUC-ROC solo con probabilidades
    auc = None
    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob)


    metrics = {
        'model': model_name,
        'experiment': experiment_name,
        'accuracy': float(accuracy),
        'sensitivity': float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1_score": float(f1),
        "auc_roc": float(auc) if auc is not None else None,
        "confusion_matrix": {
            "TP": int(tp), "FP": int(fp),
            "TN": int(tn), "FN": int(fn)
        }
    }

    print(f"\n[{model_name} - {experiment_name.upper()}] Resultados:")
    print(f"Accuracy:      {accuracy:.4f}")
    print(f"Sensibilidad:  {sensitivity:.4f}")
    print(f"Especificidad: {specificity:.4f}")
    print(f"F1-Score:      {f1:.4f}")
    if auc is not None:
        print(f"AUC-ROC:       {auc:.4f}")
    
    # Guardar
    if save_dir is None:
        save_dir = config.METRICS_DIR if config else "results/metrics"
        
    ensure_dir(save_dir)

    filename = f"metrics_{model_name.replace(' ', '')}_{experiment_name}.json"
    file_path = os.path.join(save_dir, filename)
    
    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Métricas guardadas en: {file_path}")
    
    return metrics


# Visualizacion
def plot_confusion_matrix(y_true, y_pred, model_name='Modelo', experiment_name='Exp', save_dir=None):
    """
    Genera y guarda un mapa de calor (heatmap) de la matriz de confusión.
    """
    if save_dir is None:
        save_dir = config.FIGURES_DIR if config else 'results/figures'
    ensure_dir(save_dir)

    # Matriz
    cm = confusion_matrix(y_true, y_pred)

    # Figura
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='flare',
                xticklabels=['HC (0)', 'MDD (1)'],
                yticklabels=['HC (0)', 'MDD (1)'])
    plt.title(f'Matriz de confusion: {model_name} ({experiment_name.upper()})')
    plt.ylabel('Diagnostico real')
    plt.xlabel('Prediccion')
    plt.tight_layout()

    # Guardar
    filename = f"cm_{model_name.replace(' ', '')}_{experiment_name}.png"
    file_path = os.path.join(save_dir, filename)
    plt.savefig(file_path, dpi=300)
    plt.close()

    print(f'Figura guardada en: {file_path}')
    return file_path

def plot_roc_curve(y_true, y_prob, model_name="Modelo", experiment_name="Exp", save_dir=None):
    """
    Genera y guarda la curva ROC para evaluar la capacidad de diagnóstico.
    Requiere las probabilidades (y_prob), no solo las predicciones binarias.
    """
    if y_prob is None:
        print("No se pasaron probabilidades. No se puede graficar ROC.")
        return None
        
    if save_dir is None:
        save_dir = config.FIGURES_DIR if config else "results/figures"
    ensure_dir(save_dir)

    # Calcular curva y AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    # Configurar la figura
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='pink', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='darkorange', lw=2, linestyle='--') 
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(f'Curva ROC: {model_name} ({experiment_name.upper()})')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Guardar gráfica
    filename = f"roc_{model_name.replace(' ', '')}_{experiment_name}.png"
    file_path = os.path.join(save_dir, filename)
    plt.savefig(file_path, dpi=300)
    plt.close()
    
    print(f"Curva ROC guardada en: {file_path}")
    return file_path