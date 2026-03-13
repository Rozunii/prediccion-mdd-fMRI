import os
import numpy as np
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
    X_test = np.load(os.path.join(config.PROCESSED_DIR, f'X_test_{feature}.npy'))
    y_test = np.load(os.path.join(config.PROCESSED_DIR, f'y_test_{feature}.npy'))

    svm = SVC(probability=True, random_state=config.RANDOM_STATE)

    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf'],
        'gamma': ['scale', 'auto']
    }

    # Hacer Grid
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=3)

    # Entrenamiento
    print('Buscando hiperparametros')
    grid_search.fit(X_train, y_train)

    mejor_modelo = grid_search.best_estimator_ 
    print(f'Mejores parametros: {grid_search.best_params_}')

    # Predicciones
    y_pred = mejor_modelo.predict(X_test)
    y_prob = mejor_modelo.predict_proba(X_test)[:, 1]

    # Evaluar y graficar
    print(f'Graficas y metricas')
    utils.evaluate_model(y_test, y_pred, y_prob, model_name="SVM", experiment_name=feature)
    utils.plot_confusion_matrix(y_test, y_pred, model_name="SVM", experiment_name=feature)
    utils.plot_roc_curve(y_test, y_prob, model_name="SVM", experiment_name=feature)

if __name__ == '__main__':
    experimentos = ['combined']

    for exp in experimentos:
        try:
            entrenar_evaluar_svm(exp)
        except FileNotFoundError:
            print('No se encontraron los datos')



