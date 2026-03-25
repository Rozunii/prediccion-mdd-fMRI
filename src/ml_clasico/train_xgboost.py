import os
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from src import config
    from src import utils
except ImportError:
    print('[WARN] No se encontro config.py o utils.py')
    config = None

FIGURES_DIR = os.path.join(config.FIGURES_DIR, 'xgb')
METRICS_DIR = os.path.join(config.METRICS_DIR, 'xgb')

def entrenar_evaluar_xgb(feature):
    print(f'Entrenando: {feature}')

    X_train = np.load(os.path.join(config.PROCESSED_DIR, f'X_train_{feature}.npy'))
    y_train = np.load(os.path.join(config.PROCESSED_DIR, f'y_train_{feature}.npy'))
    X_val   = np.load(os.path.join(config.PROCESSED_DIR, f'X_val_{feature}.npy'))
    y_val   = np.load(os.path.join(config.PROCESSED_DIR, f'y_val_{feature}.npy'))
    X_test  = np.load(os.path.join(config.PROCESSED_DIR, f'X_test_{feature}.npy'))
    y_test  = np.load(os.path.join(config.PROCESSED_DIR, f'y_test_{feature}.npy'))

    # scale_pos_weight compensa desbalance de clases
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos

    xgb = XGBClassifier(
        random_state=config.RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        verbosity=0
    )

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0],
    }

    # Hacer Grid
    grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='roc_auc', n_jobs=6, verbose=1)

    # Entrenamiento
    print('Buscando hiperparametros')
    grid_search.fit(X_train, y_train)

    mejor_modelo = grid_search.best_estimator_
    print(f'Mejores parametros: {grid_search.best_params_}')

    # Validacion
    y_pred_val = mejor_modelo.predict(X_val)
    y_prob_val = mejor_modelo.predict_proba(X_val)[:, 1]
    print(f'Resultados en validacion:')
    utils.evaluate_model(y_val, y_pred_val, y_prob_val, model_name="XGB", experiment_name=f'{feature}_val', save_dir=METRICS_DIR)
    utils.plot_confusion_matrix(y_val, y_pred_val, model_name="XGB", experiment_name=f'{feature}_val', save_dir=FIGURES_DIR)
    utils.plot_roc_curve(y_val, y_prob_val, model_name="XGB", experiment_name=f'{feature}_val', save_dir=FIGURES_DIR)

    # Predicciones
    y_pred = mejor_modelo.predict(X_test)
    y_prob = mejor_modelo.predict_proba(X_test)[:, 1]

    # Evaluar y graficar
    print(f'Graficas y metricas')
    utils.evaluate_model(y_test, y_pred, y_prob, model_name="XGB", experiment_name=feature, save_dir=METRICS_DIR)
    utils.plot_confusion_matrix(y_test, y_pred, model_name="XGB", experiment_name=feature, save_dir=FIGURES_DIR)
    utils.plot_roc_curve(y_test, y_prob, model_name="XGB", experiment_name=feature, save_dir=FIGURES_DIR)

    # Guardar modelo
    model_path = os.path.join(config.MODELS_DIR, f'xgb_{feature}.pkl')
    joblib.dump(mejor_modelo, model_path)
    print(f'Modelo guardado: {model_path}')

if __name__ == '__main__':
    use_combat = len(sys.argv) > 1 and sys.argv[1].lower() == 'combat'
    sufijo = '_combat' if use_combat else ''

    experimentos = ['fc', 'alff', 'combined', 'fc_anova', 'combined_anova', 'fc_mrmr', 'combined_mrmr']

    for exp in experimentos:
        feature = f'{exp}{sufijo}'
        metrics_file = os.path.join(METRICS_DIR, f'metrics_XGB_{feature}.json')
        if os.path.exists(metrics_file):
            print(f'[SKIP] Ya existe: {metrics_file}')
            continue
        try:
            entrenar_evaluar_xgb(feature)
        except FileNotFoundError as e:
            print(f'No se encontraron los datos para {exp}: {e}')