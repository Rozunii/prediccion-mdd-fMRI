"""
data_loader.py -- Carga y preprocesamiento del dataset PHQ-9

Dataset: 14-day Ambulatory Assessment of Depression Symptoms
  - 185 pacientes, 16150 observaciones
  - Cada fila = una evaluacion en un momento del tiempo
  - Variables: items PHQ-9, felicidad, demograficos, contexto temporal

Pasos:
  1. Cargar CSV y calcular phq_total como target
  2. Descartar columnas con >80% nulos (q1-q47)
  3. Imputar age con mediana por user_id
  4. Encodear variables categoricas (sex, period.name)
  5. Extraer hora del dia desde columna time
  6. Split por user_id para evitar leakage entre train y test

Proyecto: Prediccion de depresion con IA -- Samsung Innovation Campus
Dataset: PHQ-9 14-day Ambulatory Assessment
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from src import config
except ImportError:
    print('[WARN] No se encontro config.py')
    config = None

# Ruta al CSV
PHQ9_FILE = os.path.join(
    str(config.RAW_DIR) if config else 'data/raw',
    'PHQ-9',
    'Dataset_14-day_AA_depression_symptoms_mood_and_PHQ-9.csv'
)

# Columnas de los 9 items del cuestionario
PHQ_ITEMS = [f'phq{i}' for i in range(1, 10)]

# Features para regresion
REGRESSION_FEATURES = ['happiness.score', 'phq.day', 'hour', 'period.name', 'sex', 'age']

# Features para clustering
CLUSTERING_FEATURES = PHQ_ITEMS + ['happiness.score']


def cargar_datos(filepath: str = None) -> pd.DataFrame:
    """Carga el CSV y aplica limpieza inicial.

    Args:
        filepath: ruta al CSV (usa PHQ9_FILE por defecto)

    Returns:
        DataFrame limpio con phq_total calculado y columnas inutiles eliminadas
    """
    if filepath is None:
        filepath = PHQ9_FILE

    print(f'[INFO] Cargando: {filepath}')
    df = pd.read_csv(filepath)
    print(f'[INFO] Shape original: {df.shape}')

    # Eliminar columnas inutiles o redundantes
    # Unnamed:0 e id son indices, start.time es redundante con time
    cols_drop = ['Unnamed: 0', 'id', 'start.time']

    # Columnas q1-q47: >88% nulos, sin valor practico
    cols_q = [c for c in df.columns if c.startswith('q') and c not in ['q1']]
    cols_drop += [c for c in cols_q if c in df.columns]

    df = df.drop(columns=cols_drop, errors='ignore')

    # Calcular target: score total PHQ-9
    df['phq_total'] = df[PHQ_ITEMS].sum(axis=1, min_count=1)

    # Eliminar filas sin target (todos los items nulos)
    n_antes = len(df)
    df = df.dropna(subset=['phq_total'])
    print(f'[INFO] Filas eliminadas sin target: {n_antes - len(df)}')
    print(f'[INFO] Shape tras limpieza: {df.shape}')

    return df


def preprocesar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transformaciones:
      - age: imputa nulos con mediana por user_id
      - sex: label encoding (female=0, male=1, transgender=2)
      - period.name: ordinal encoding (morning=0, midday=1, evening=2)
      - time: extrae hora del dia como entero (0-23)

    Args:
        df: DataFrame de cargar_datos()

    Returns:
        DataFrame con features transformadas listo para ML
    """
    df = df.copy()

    # Extraer hora desde columna time (es un timestamp completo)
    df['hour'] = pd.to_datetime(df['time'], errors='coerce').dt.hour
    df = df.drop(columns=['time'])

    # Imputar age: mediana por user_id
    age_mediana = df.groupby('user_id')['age'].transform('median')
    df['age'] = df['age'].fillna(age_mediana)
    df['age'] = df['age'].fillna(df['age'].median())

    # Encoding sex
    sex_map = {'female': 0, 'male': 1, 'transgender': 2}
    df['sex'] = df['sex'].map(sex_map)
    # Nulos de sex -> moda del dataset
    df['sex'] = df['sex'].fillna(df['sex'].mode()[0])

    # Encoding period.name
    period_map = {'morning': 0, 'midday': 1, 'evening': 2}
    df['period.name'] = df['period.name'].map(period_map)
    df['period.name'] = df['period.name'].fillna(df['period.name'].mode()[0])

    print('[INFO] Features preprocesadas:')
    print(f'  Nulos restantes:\n{df[REGRESSION_FEATURES].isnull().sum()}')

    return df


def hacer_split(df: pd.DataFrame,
                test_size: float = 0.15,
                val_size: float = 0.15,
                random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide el dataset en train/val/test por user_id.

    Todos los registros de un mismo paciente van al mismo split
    para evitar leakage entre conjuntos.

    Args:
        df:           DataFrame preprocesado
        test_size:    proporcion de usuarios para test
        val_size:     proporcion de usuarios para validacion
        random_state: semilla para reproducibilidad

    Returns:
        Tupla (df_train, df_val, df_test)
    """
    usuarios = df['user_id'].unique()

    # Primer split: separar test
    users_temp, users_test = train_test_split(
        usuarios,
        test_size=test_size,
        random_state=random_state
    )

    # Segundo split: separar val del restante
    val_relative = val_size / (1 - test_size)
    users_train, users_val = train_test_split(
        users_temp,
        test_size=val_relative,
        random_state=random_state
    )

    df_train = df[df['user_id'].isin(users_train)].reset_index(drop=True)
    df_val   = df[df['user_id'].isin(users_val)].reset_index(drop=True)
    df_test  = df[df['user_id'].isin(users_test)].reset_index(drop=True)

    print(f'[INFO] Split por user_id (70/15/15):')
    print(f'  Train: {len(users_train)} usuarios, {len(df_train)} filas')
    print(f'  Val:   {len(users_val)} usuarios, {len(df_val)} filas')
    print(f'  Test:  {len(users_test)} usuarios, {len(df_test)} filas')

    return df_train, df_val, df_test


def preparar_regresion(df_train: pd.DataFrame,
                       df_val: pd.DataFrame,
                       df_test: pd.DataFrame,
                       save_dir: str = None) -> dict:
    """Escala features y separa X/y para regresion.

    StandardScaler fit solo en train, aplicado a val y test.

    Args:
        df_train: DataFrame de entrenamiento
        df_val:   DataFrame de validacion
        df_test:  DataFrame de prueba
        save_dir: carpeta donde guardar el scaler .pkl

    Returns:
        Diccionario con X_train, X_val, X_test, y_train, y_val, y_test
    """
    if save_dir is None:
        save_dir = str(config.MODELS_DIR / 'phq9') if config else 'results/models/phq9'
    os.makedirs(save_dir, exist_ok=True)

    X_train = df_train[REGRESSION_FEATURES].values
    X_val   = df_val[REGRESSION_FEATURES].values
    X_test  = df_test[REGRESSION_FEATURES].values

    y_train = df_train['phq_total'].values
    y_val   = df_val['phq_total'].values
    y_test  = df_test['phq_total'].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    joblib.dump(scaler, os.path.join(save_dir, 'scaler_regression.pkl'))
    print(f'[INFO] Scaler guardado: {save_dir}/scaler_regression.pkl')

    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'feature_names': REGRESSION_FEATURES,
    }


def preparar_clustering(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Agrega por user_id y prepara matriz para clustering.

    Cada usuario queda representado por:
      - Media de cada item phq1-phq9 (perfil de sintomas)
      - Media de happiness.score
      - Desviacion estandar de phq_total (variabilidad temporal)

    Args:
        df: DataFrame completo preprocesado

    Returns:
        Tupla (X_cluster, user_ids) donde X_cluster es (n_usuarios, n_features)
    """
    # PHQ items son fijos por usuario (evaluacion baseline)
    # happiness.score varia cada dia -> usamos media Y std como features
    agg_dict = {item: 'first' for item in PHQ_ITEMS}
    agg_dict['happiness.score'] = ['mean', 'std']
    agg_dict['phq_total'] = 'first'
    agg_dict['age'] = 'first'
    agg_dict['sex'] = 'first'

    df_agg = df.groupby('user_id').agg(agg_dict)

    # Aplanar columnas multi-nivel
    df_agg.columns = [
        f'{col[0]}_{col[1]}' if col[1] else col[0]
        for col in df_agg.columns
    ]
    df_agg = df_agg.rename(columns={
        'happiness.score_mean': 'happiness_mean',
        'happiness.score_std':  'happiness_std',
        'phq_total_first':      'phq_total_mean',
        'age_first': 'age',
        'sex_first': 'sex',
    })

    # Renombrar items phq (first -> nombre limpio)
    for item in PHQ_ITEMS:
        df_agg = df_agg.rename(columns={f'{item}_first': item})

    # Rellenar std nulos (usuarios con una sola observacion de happiness)
    df_agg['happiness_std'] = df_agg['happiness_std'].fillna(0)

    # Imputar NaN restantes con mediana de cada columna
    cluster_cols = PHQ_ITEMS + ['happiness_mean', 'happiness_std']
    for col in cluster_cols:
        if df_agg[col].isnull().any():
            mediana = df_agg[col].median()
            df_agg[col] = df_agg[col].fillna(mediana)
            print(f'  [WARN] NaN en {col} imputados con mediana={mediana:.2f}')

    n_nulos = df_agg[cluster_cols].isnull().sum().sum()
    if n_nulos > 0:
        print(f'  [ERROR] Siguen habiendo {n_nulos} NaN tras imputacion')

    user_ids = df_agg.index.values

    X_cluster = df_agg[cluster_cols].values

    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(X_cluster)

    print(f'[INFO] Matriz clustering: {X_cluster.shape}')
    print(f'  Features: {cluster_cols}')

    return X_cluster, user_ids, df_agg


if __name__ == '__main__':
    # Pipeline completo de preprocesamiento
    df_raw  = cargar_datos()
    df      = preprocesar_features(df_raw)

    df_train, df_val, df_test = hacer_split(df)

    print('\n[INFO] Preparando datos para regresion...')
    splits_reg = preparar_regresion(df_train, df_val, df_test)
    print(f'  X_train: {splits_reg["X_train"].shape}')
    print(f'  y_train: media={splits_reg["y_train"].mean():.2f}, std={splits_reg["y_train"].std():.2f}')

    print('\n[INFO] Preparando datos para clustering...')
    X_cluster, user_ids, df_agg = preparar_clustering(df)
    print(f'  Usuarios: {len(user_ids)}')
    print(f'  PHQ total medio por usuario: {df_agg["phq_total_mean"].mean():.2f}')

    print('\n[INFO] data_loader.py completado.')
