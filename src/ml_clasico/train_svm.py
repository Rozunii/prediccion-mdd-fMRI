import os
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import sys
 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
 
try:
    from src import config
    from src import utils
except ImportError:
    print('[WARN] No se encontro config.py o utils.py')
    config = None
 
def entrenar_evaluar_svm(feature):
    print(f'Entrenando: {feature}')
 
    X_train = np.load(os.path.join(config.PROCESSED_DIR, f'X_train_{feature}.npy'))
    y_train = np.load(os.path.join(config.PROCESSED_DIR, f'y_train_{feature}.npy'))
    X_val   = np.load(os.path.join(config.PROCESSED_DIR, f'X_val_{feature}.npy'))
    y_val   = np.load(os.path.join(config.PROCESSED_DIR, f'y_val_{feature}.npy'))
    X_test  = np.load(os.path.join(config.PROCESSED_DIR, f'X_test_{feature}.npy'))
    y_test  = np.load(os.path.join(config.PROCESSED_DIR, f'y_test_{feature}.npy'))
 
    svm = SVC(probability=True, random_state=config.RANDOM_STATE)
 
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf'],
        'gamma': ['scale', 'auto']
    }
 
    # Hacer Grid
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
 
    # Entrenamiento
    print('Buscando hiperparametros')
    grid_search.fit(X_train, y_train)
 
    mejor_modelo = grid_search.best_estimator_
    print(f'Mejores parametros: {grid_search.best_params_}')
 
    # Validacion
    y_pred_val = mejor_modelo.predict(X_val)
    y_prob_val = mejor_modelo.predict_proba(X_val)[:, 1]
    print(f'Resultados en validacion:')
    utils.evaluate_model(y_val, y_pred_val, y_prob_val, model_name="SVM", experiment_name=f'{feature}_val')
 
    # Predicciones
    y_pred = mejor_modelo.predict(X_test)
    y_prob = mejor_modelo.predict_proba(X_test)[:, 1]
 
    # Evaluar y graficar
    print(f'Graficas y metricas')
    utils.evaluate_model(y_test, y_pred, y_prob, model_name="SVM", experiment_name=feature)
    utils.plot_confusion_matrix(y_test, y_pred, model_name="SVM", experiment_name=feature)
    utils.plot_roc_curve(y_test, y_prob, model_name="SVM", experiment_name=feature)
 
    # Guardar modelo
    model_path = os.path.join(config.MODELS_DIR, f'svm_{feature}.pkl')
    joblib.dump(mejor_modelo, model_path)
    print(f'Modelo guardado: {model_path}')
 
if __name__ == '__main__':
    use_combat = len(sys.argv) > 1 and sys.argv[1].lower() == 'combat'
    sufijo = '_combat' if use_combat else ''
 
    experimentos = ['fc', 'alff', 'combined', 'fc_anova', 'combined_anova']
 
    for exp in experimentos:
        try:
            entrenar_evaluar_svm(f'{exp}{sufijo}')
        except FileNotFoundError as e:
            print(f'No se encontraron los datos para {exp}: {e}')



