"""
train_mlp.py -- MLP para clasificacion MDD/HC con datos de fMRI

Entrena un MLP sobre los splits preprocesados por preprocessing.py.
Usa los mismos datos que los modelos clasicos para comparacion directa.

Experimentos:
  - combined_combat      
  - combined_mrmr_combat 

Arquitectura:
  Input(n) -> Dense(256) -> ReLU -> Dropout(0.3)
           -> Dense(128) -> ReLU -> Dropout(0.3)
           -> Dense(64)  -> ReLU
           -> Dense(1)   -> Sigmoid
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from src import config, utils
except ImportError:
    print('[WARN] No se encontro config.py')
    config = None

FIGURES_DIR = os.path.join(str(config.FIGURES_DIR) if config else 'results/figures', 'mlp')
METRICS_DIR = os.path.join(str(config.METRICS_DIR) if config else 'results/metrics', 'mlp')
MODELS_DIR  = os.path.join(str(config.MODELS_DIR)  if config else 'results/models',  'mlp')

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

BATCH_SIZE = 64
LR         = 1e-3
MAX_EPOCHS = 200
PATIENCE   = 20
DROPOUT    = 0.3

tf.random.set_seed(42)
np.random.seed(42)


def cargar_splits(experimento: str) -> dict:
    """Carga los arrays preprocesados de un experimento desde disco.

    Args:
        experimento

    Returns:
        Diccionario con X_train, X_val, X_test, y_train, y_val, y_test
    """
    processed_dir = str(config.PROCESSED_DIR) if config else 'data/processed'
    splits = {}
    for split in ['train', 'val', 'test']:
        splits[f'X_{split}'] = np.load(
            os.path.join(processed_dir, f'X_{split}_{experimento}.npy')
        )
        splits[f'y_{split}'] = np.load(
            os.path.join(processed_dir, f'y_{split}_{experimento}.npy')
        )
    return splits


def construir_modelo(input_dim: int) -> tf.keras.Model:
    """Construye el MLP con la API Sequential de Keras.

    Args:
        input_dim

    Returns:
        Modelo Keras compilado
    """
    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )
    return modelo


def graficar_curvas(historia: tf.keras.callbacks.History, experimento: str) -> None:
    """Grafica loss y accuracy de train/val durante el entrenamiento.

    Args:
        historia
        experimento
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(historia.history['loss'], label='Train', color='steelblue')
    if 'val_loss' in historia.history:
        ax1.plot(historia.history['val_loss'], label='Val', color='coral')
    ax1.set_xlabel('Epoca')
    ax1.set_ylabel('BCE Loss')
    ax1.set_title('Loss')
    ax1.legend()

    ax2.plot(historia.history['auc'], label='Train', color='steelblue')
    if 'val_auc' in historia.history:
        ax2.plot(historia.history['val_auc'], label='Val', color='coral')
    ax2.set_xlabel('Epoca')
    ax2.set_ylabel('AUC-ROC')
    ax2.set_title('AUC-ROC')
    ax2.legend()

    plt.suptitle(f'MLP — {experimento}')
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f'curvas_{experimento}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'[INFO] Curvas guardadas: {path}')


def evaluar_y_guardar(modelo: tf.keras.Model, splits: dict, experimento: str) -> None:
    """Calcula metricas en val y test, guarda JSON y figuras.

    Args:
        modelo
        splits
        experimento
    """
    for split_name in [k.replace('X_', '') for k in splits.keys() if k.startswith('X_')]:
        X      = splits[f'X_{split_name}']
        y_true = splits[f'y_{split_name}']

        y_prob = modelo.predict(X, verbose=0).flatten()
        y_pred = (y_prob >= 0.5).astype(int)

        auc  = roc_auc_score(y_true, y_prob)
        acc  = accuracy_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        f1   = 2 * tp / (2 * tp + fp + fn)

        metricas = {
            'model':       'MLP',
            'experiment':  experimento,
            'split':       split_name,
            'auc_roc':     round(float(auc),  4),
            'accuracy':    round(float(acc),  4),
            'sensitivity': round(float(sens), 4),
            'specificity': round(float(spec), 4),
            'f1_score':    round(float(f1),   4),
        }

        sufijo = '_val' if split_name == 'val' else ''
        path = os.path.join(METRICS_DIR, f'metrics_MLP_{experimento}{sufijo}.json')
        with open(path, 'w') as f:
            json.dump(metricas, f, indent=2)

        print(f'[{split_name.upper()}] AUC={auc:.4f} | Acc={acc:.4f} | '
              f'Sens={sens:.4f} | Spec={spec:.4f} | F1={f1:.4f}')

        if config:
            utils.plot_confusion_matrix(y_true, y_pred, model_name='MLP',
                                        experiment_name=f'{experimento}{sufijo}',
                                        save_dir=FIGURES_DIR)
            utils.plot_roc_curve(y_true, y_prob, model_name='MLP',
                                 experiment_name=f'{experimento}{sufijo}',
                                 save_dir=FIGURES_DIR)


def entrenar_mlp(experimento: str, k_folds: int = 5) -> None:
    """Pipeline con K-Fold CV: selecciona hiperparametros sin sesgar el val set.

    Flujo:
      1. Combinar train+val en un solo conjunto de desarrollo
      2. K-Fold estratificado sobre desarrollo -> obtener AUC promedio por fold
      3. Entrenar modelo final sobre desarrollo completo
      4. Evaluar modelo final en test (datos nunca vistos)

    Args:
        experimento: nombre del experimento (ej. 'combined_combat')
        k_folds:     numero de folds para cross-validation
    """
    metrics_path = os.path.join(METRICS_DIR, f'metrics_MLP_{experimento}.json')
    if os.path.exists(metrics_path):
        print(f'[SKIP] {experimento} ya tiene metricas')
        return

    print(f'[INFO] Entrenando: {experimento} ({k_folds}-Fold CV)')

    splits    = cargar_splits(experimento)
    input_dim = splits['X_train'].shape[1]

    # Combinar train+val para el desarrollo (test permanece intacto)
    X_dev = np.concatenate([splits['X_train'], splits['X_val']], axis=0)
    y_dev = np.concatenate([splits['y_train'], splits['y_val']], axis=0)
    X_test, y_test = splits['X_test'], splits['y_test']

    print(f'[INFO] Input: {input_dim} features | Dev: {len(y_dev)} | Test: {len(y_test)}')

    n_pos = y_dev.sum()
    n_neg = len(y_dev) - n_pos
    class_weight = {0: 1.0, 1: float(n_neg / n_pos)}
    print(f'[INFO] Class weight MDD: {class_weight[1]:.3f}')

    # K-Fold estratificado: mantiene proporcion MDD/HC en cada fold
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    aucs_fold   = []
    epocas_fold = []

    for fold, (idx_tr, idx_val) in enumerate(skf.split(X_dev, y_dev), start=1):
        print(f'[INFO] Fold {fold}/{k_folds}')
        X_tr, X_val = X_dev[idx_tr], X_dev[idx_val]
        y_tr, y_val = y_dev[idx_tr], y_dev[idx_val]

        modelo_fold = construir_modelo(input_dim)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc', mode='max', patience=PATIENCE,
                restore_best_weights=True, verbose=0
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_auc', mode='max', factor=0.5, patience=8, verbose=0
            ),
        ]

        modelo_fold.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=0,
        )

        epocas_fold.append(len(modelo_fold.history.history['loss']))
        y_prob_val = modelo_fold.predict(X_val, verbose=0).flatten()
        auc_fold   = roc_auc_score(y_val, y_prob_val)
        aucs_fold.append(auc_fold)
        print(f'  Fold {fold} AUC val: {auc_fold:.4f} | Epocas: {epocas_fold[-1]}')

    print(f'[INFO] AUC CV: {np.mean(aucs_fold):.4f} +/- {np.std(aucs_fold):.4f}')

    # Modelo final: entrena por el promedio de epocas de los folds
    epocas_final = int(np.mean(epocas_fold))
    print(f'[INFO] Entrenando modelo final sobre dev completo ({epocas_final} epocas)...')
    modelo_final = construir_modelo(input_dim)

    historia = modelo_final.fit(
        X_dev, y_dev,
        epochs=epocas_final,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        verbose=1,
    )

    graficar_curvas(historia, experimento)

    # Evaluar solo en test (val seria sobre datos de entrenamiento, no tiene sentido)
    splits_final = {
        'X_test': X_test, 'y_test': y_test,
    }
    evaluar_y_guardar(modelo_final, splits_final, experimento)

    # Guardar AUC de CV en el JSON de metricas
    with open(metrics_path) as f:
        metricas = json.load(f)
    metricas['cv_auc_mean'] = round(float(np.mean(aucs_fold)), 4)
    metricas['cv_auc_std']  = round(float(np.std(aucs_fold)),  4)
    with open(metrics_path, 'w') as f:
        json.dump(metricas, f, indent=2)

    modelo_final.save(os.path.join(MODELS_DIR, f'mlp_{experimento}.keras'))
    print(f'[INFO] Modelo guardado: {MODELS_DIR}/mlp_{experimento}.keras')


if __name__ == '__main__':
    experimentos = ['combined_combat', 'combined_mrmr_combat']
    for exp in experimentos:
        entrenar_mlp(exp)

    print('\n[INFO] Resumen final MLP')
    print(f'{"Experimento":<26} {"AUC test":>9} {"AUC CV":>9} {"Sens":>6} {"Spec":>6} {"F1":>6}')
    for exp in experimentos:
        path = os.path.join(METRICS_DIR, f'metrics_MLP_{exp}.json')
        if os.path.exists(path):
            m = json.load(open(path))
            cv = f'{m["cv_auc_mean"]:.3f}+/-{m["cv_auc_std"]:.3f}' if 'cv_auc_mean' in m else '---'
            print(f'{exp:<26} {m["auc_roc"]:>9.3f} {cv:>9} {m["sensitivity"]:>6.3f} '
                  f'{m["specificity"]:>6.3f} {m["f1_score"]:>6.3f}')
