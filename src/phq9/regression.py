"""
regression.py -- Regresion sobre el score PHQ-9 total

Predice la severidad de depresion (phq_total, 0-27) a partir de
variables no clinicas: felicidad, dia del estudio, hora, periodo,
sexo y edad.

Modelos comparados:
  - Ridge        
  - SVR          
  - XGBoost      

Metricas:
  - RMSE  
  - MAE   
  - R2    
  - PCC   

"""

import os
import sys
import json
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from src import config
    from src.phq9.data_loader import cargar_datos, preprocesar_features, hacer_split, preparar_regresion
except ImportError:
    print('[WARN] No se encontro config.py')
    config = None

FIGURES_DIR = os.path.join(str(config.FIGURES_DIR) if config else 'results/figures', 'phq9')
METRICS_DIR = os.path.join(str(config.METRICS_DIR) if config else 'results/metrics', 'phq9')
MODELS_DIR  = os.path.join(str(config.MODELS_DIR)  if config else 'results/models',  'phq9')

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)


def evaluar_regresor(y_true: np.ndarray, y_pred: np.ndarray,
                     nombre: str, split: str) -> dict:
    """Calcula metricas de regresion y las guarda en JSON.

    Args:
        y_true
        y_pred
        nombre
        split

    Returns:
        Diccionario con rmse, mae, r2, pcc
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    pcc, _ = pearsonr(y_true, y_pred)

    metricas = {
        'model':     nombre,
        'split':     split,
        'rmse':      round(float(rmse), 4),
        'mae':       round(float(mae),  4),
        'r2':        round(float(r2),   4),
        'pearson_r': round(float(pcc),  4),
    }

    path = os.path.join(METRICS_DIR, f'metrics_{nombre}_{split}.json')
    with open(path, 'w') as f:
        json.dump(metricas, f, indent=2)

    print(f'  [{split.upper()}] RMSE={rmse:.3f} | MAE={mae:.3f} | R2={r2:.3f} | PCC={pcc:.3f}')
    return metricas


def graficar_predicciones(y_true: np.ndarray, y_pred: np.ndarray, nombre: str) -> None:
    """Scatter plot de valores reales vs predichos.

    Args:
        y_true
        y_pred
        nombre
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color='steelblue')

    # Linea perfecta
    lims = [0, 27]
    ax.plot(lims, lims, 'r--', linewidth=1, label='Prediccion perfecta')

    ax.set_xlabel('PHQ-9 real')
    ax.set_ylabel('PHQ-9 predicho')
    ax.set_title(f'{nombre} — Real vs Predicho (test)')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend()
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, f'scatter_{nombre}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Figura guardada: {path}')


def graficar_feature_importance(modelo, feature_names: list, nombre: str) -> None:
    """Grafica feature importance.

    Args:
        modelo
        feature_names
        nombre
    """
    importances = modelo.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(importances)), importances[indices], color='steelblue')
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=30, ha='right')
    ax.set_title('XGBoost — Feature Importance')
    ax.set_ylabel('Importancia')
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, 'feature_importance_XGB.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Figura guardada: {path}')


def entrenar_ridge(splits: dict) -> dict:
    """Entrena Ridge con GridSearchCV y evalua en val y test.

    Args:
        splits: dict con X_train, X_val, X_test, y_train, y_val, y_test

    Returns:
        Diccionario con metricas de val y test
    """
    print('\n[Ridge]')

    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    gs = GridSearchCV(Ridge(), param_grid, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1)
    gs.fit(splits['X_train'], splits['y_train'])

    print(f'  Mejor alpha: {gs.best_params_["alpha"]}')
    modelo = gs.best_estimator_

    y_pred_val  = modelo.predict(splits['X_val'])
    y_pred_test = modelo.predict(splits['X_test'])

    m_val  = evaluar_regresor(splits['y_val'],  y_pred_val,  'Ridge', 'val')
    m_test = evaluar_regresor(splits['y_test'], y_pred_test, 'Ridge', 'test')
    graficar_predicciones(splits['y_test'], y_pred_test, 'Ridge')

    joblib.dump(modelo, os.path.join(MODELS_DIR, 'Ridge.pkl'))
    return {'val': m_val, 'test': m_test}


def entrenar_svr(splits: dict) -> dict:
    """Entrena SVR con GridSearchCV y evalua en val y test.

    Args:
        splits: dict con X_train, X_val, X_test, y_train, y_val, y_test

    Returns:
        Diccionario con metricas de val y test
    """
    print('\n[SVR]')

    param_grid = {
        'C':       [0.1, 1.0, 10.0],
        'epsilon': [0.5, 1.0, 2.0],
        'kernel':  ['rbf'],
    }
    gs = GridSearchCV(SVR(), param_grid, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1)
    gs.fit(splits['X_train'], splits['y_train'])

    print(f'  Mejores params: {gs.best_params_}')
    modelo = gs.best_estimator_

    y_pred_val  = modelo.predict(splits['X_val'])
    y_pred_test = modelo.predict(splits['X_test'])

    m_val  = evaluar_regresor(splits['y_val'],  y_pred_val,  'SVR', 'val')
    m_test = evaluar_regresor(splits['y_test'], y_pred_test, 'SVR', 'test')
    graficar_predicciones(splits['y_test'], y_pred_test, 'SVR')

    joblib.dump(modelo, os.path.join(MODELS_DIR, 'SVR.pkl'))
    return {'val': m_val, 'test': m_test}


def entrenar_xgb(splits: dict) -> dict:
    """Entrena XGBoost Regressor con GridSearchCV y evalua en val y test.

    Args:
        splits: dict con X_train, X_val, X_test, y_train, y_val, y_test

    Returns:
        Diccionario con metricas de val y test
    """
    print('\n[XGBoost]')

    param_grid = {
        'n_estimators': [100, 300],
        'max_depth':    [3, 5],
        'learning_rate':[0.05, 0.1],
        'subsample':    [0.8],
    }
    gs = GridSearchCV(
        XGBRegressor(random_state=42, n_jobs=2, verbosity=0),
        param_grid,
        scoring='neg_root_mean_squared_error',
        cv=5,
        n_jobs=1
    )
    gs.fit(splits['X_train'], splits['y_train'])

    print(f'  Mejores params: {gs.best_params_}')
    modelo = gs.best_estimator_

    y_pred_val  = modelo.predict(splits['X_val'])
    y_pred_test = modelo.predict(splits['X_test'])

    m_val  = evaluar_regresor(splits['y_val'],  y_pred_val,  'XGB', 'val')
    m_test = evaluar_regresor(splits['y_test'], y_pred_test, 'XGB', 'test')
    graficar_predicciones(splits['y_test'], y_pred_test, 'XGB')
    graficar_feature_importance(modelo, splits['feature_names'], 'XGB')

    joblib.dump(modelo, os.path.join(MODELS_DIR, 'XGB.pkl'))
    return {'val': m_val, 'test': m_test}


def resumen_modelos() -> None:
    """Lee los JSON de metricas y muestra tabla comparativa."""
    import glob

    archivos = glob.glob(os.path.join(METRICS_DIR, 'metrics_*_test.json'))
    if not archivos:
        print('[WARN] No se encontraron metricas de test.')
        return

    print('\n Resumen')
    print(f'{"Modelo":<10} {"RMSE":>7} {"MAE":>7} {"R2":>7} {"PCC":>7}')
    print('-' * 38)
    for f in sorted(archivos):
        with open(f) as fh:
            m = json.load(fh)
        print(f'{m["model"]:<10} {m["rmse"]:>7.3f} {m["mae"]:>7.3f} {m["r2"]:>7.3f} {m["pearson_r"]:>7.3f}')


if __name__ == '__main__':
    # Pipeline completo
    df_raw = cargar_datos()
    df     = preprocesar_features(df_raw)
    df_train, df_val, df_test = hacer_split(df)
    splits = preparar_regresion(df_train, df_val, df_test)

    entrenar_ridge(splits)
    entrenar_svr(splits)
    entrenar_xgb(splits)

    resumen_modelos()
