"""
preprocessing.py — Preprocesamiento de features para ML clásico

Pasos:
  1. Cargar features y labels desde data/processed/
  2. Split 70/15/15 estratificado (mismo para los 3 experimentos)
  3. StandardScaler (fit en train, transform en val/test)
  4. PCA para FC y Combinado (ALFF con 49 features no lo necesita)
  5. Guardar preprocesadores como .pkl para reusar

Funciones:
  - load_processed_data()  : Carga los .npy de data/processed/
  - make_splits()          : Divide en train/val/test estratificado
  - preprocess_experiment() : Escala y aplica PCA a un experimento
  - preprocess_all()       : Pipeline completo para los 3 experimentos

Proyecto: Predicción de depresión con IA — Samsung Innovation Campus
Dataset: REST-meta-MDD Phase 1
"""

import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.pipeline import Pipeline

# Importar configuración
try:
    import config
except ImportError:
    print("[WARN] No se encontró config.py.")
    config = None


# 1. Carga de datos

def load_processed_data(processed_dir=None):
    """
    Carga los archivos .npy generados por feature_extraction.py.
    
    Estos archivos ya tienen los features extraídos.
    
    Args:
        processed_dir: ruta a data/processed/ (usa config si no se da)
    
    Returns:
        data: dict con keys:
            'fc'       : array (2293, 1679028) — features de conectividad
            'alff'     : array (2293, 49)      — features ALFF
            'combined' : array (2293, 1679077) — FC + ALFF concatenados
            'labels'   : array (2293,)         — 1=MDD, 0=HC
            'ids'      : array (2293,)         — subject IDs
    """
    if processed_dir is None:
        if config is not None:
            processed_dir = config.PROCESSED_DIR
        else:
            raise ValueError("Proporciona processed_dir o configura config.py")

    print(f"[INFO] Cargando datos desde: {processed_dir}")

    data = {
        'fc':       np.load(os.path.join(processed_dir, 'fc_features.npy'), mmap_mode='r'),
        'alff':     np.load(os.path.join(processed_dir, 'alff_features.npy')),
        'combined': np.load(os.path.join(processed_dir, 'combined_features.npy'), mmap_mode='r'),
        'labels':   np.load(os.path.join(processed_dir, 'labels.npy')),
        'ids':      np.load(os.path.join(processed_dir, 'subject_ids.npy'),
                            allow_pickle=True),
    }

    print(f"[INFO] Datos cargados:")
    print(f"  FC:       {data['fc'].shape}")
    print(f"  ALFF:     {data['alff'].shape}")
    print(f"  Combined: {data['combined'].shape}")
    print(f"  Labels:   {data['labels'].shape} "
          f"({data['labels'].sum()} MDD, {(data['labels'] == 0).sum()} HC)")

    return data



# 2. Split Train / Val / Test


def make_splits(labels, test_size=None, val_size=None, random_state=None):
    """
    Genera los ÍNDICES para dividir en train/val/test.
    
    Devuelve solo índices (no los datos) para que el mismo split
    se aplique a los 3 experimentos (FC, ALFF, combinado).
    Así la comparación entre experimentos es justa: los mismos
    sujetos están en train, val y test en los 3 casos.
    
    Split por defecto: 70% train, 15% val, 15% test
    Estratificado: misma proporción MDD/HC en cada set.
    
    Args:
        labels: array de shape (n_sujetos,) con 0s y 1s
        test_size: proporción de test (default de config: 0.15)
        val_size: proporción de val (default de config: 0.15)
        random_state: semilla (default de config: 42)
    
    Returns:
        splits: dict con keys 'train_idx', 'val_idx', 'test_idx'
                cada uno es un array de índices
    """
    # Usar valores de config si no se pasan
    if test_size is None:
        test_size = config.TEST_SIZE if config else 0.15
    if val_size is None:
        val_size = config.VAL_SIZE if config else 0.15
    if random_state is None:
        random_state = config.RANDOM_STATE if config else 42

    n_total = len(labels)
    indices = np.arange(n_total)

    # Primer split: separar test (15%)
    idx_temp, idx_test = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    # Segundo split: separar val del restante
    # val_size es 15% del total, pero ahora tenemos 85% del total
    # Entonces: val_relative = 0.15 / 0.85 ≈ 0.176
    val_relative = val_size / (1 - test_size)

    idx_train, idx_val = train_test_split(
        idx_temp,
        test_size=val_relative,
        random_state=random_state,
        stratify=labels[idx_temp]
    )

    splits = {
        'train_idx': idx_train,
        'val_idx':   idx_val,
        'test_idx':  idx_test,
    }

    # Reportar
    y_train = labels[idx_train]
    y_val = labels[idx_val]
    y_test = labels[idx_test]

    print(f"[INFO] Split completado:")
    print(f"  Train: {len(idx_train)} ({y_train.sum()} MDD, "
          f"{(y_train == 0).sum()} HC)")
    print(f"  Val:   {len(idx_val)} ({y_val.sum()} MDD, "
          f"{(y_val == 0).sum()} HC)")
    print(f"  Test:  {len(idx_test)} ({y_test.sum()} MDD, "
          f"{(y_test == 0).sum()} HC)")

    return splits


# 3. Escalado y PCA


def preprocess_experiment(X, labels, splits, experiment_name,
                          apply_pca=True, n_components=None,
                          save_dir=None):
    """
    Aplica StandardScaler y opcionalmente PCA a un experimento.
    
    Importante: fit solo en train, transform en val y test.
    Esto evita data leakage (que info de val/test se filtre al train).
    
    Usa sklearn Pipeline para encadenar Scaler + PCA en un solo objeto
    que se puede guardar y reusar después.
    
    Args:
        X: array de features (n_sujetos, n_features)
        labels: array de labels (n_sujetos,)
        splits: dict de make_splits() con los índices
        experiment_name: nombre del experimento ('fc', 'alff', 'combined')
        apply_pca: si aplicar PCA (False para ALFF que solo tiene 49 features)
        n_components: componentes PCA (default de config: 100)
        save_dir: carpeta donde guardar el preprocesador .pkl
    
    Returns:
        result: dict con:
            'X_train', 'X_val', 'X_test': arrays preprocesados
            'y_train', 'y_val', 'y_test': arrays de labels
            'pipeline': objeto sklearn Pipeline (scaler + pca)
    """
    if n_components is None:
        n_components = config.PCA_COMPONENTS if config else 100
    if save_dir is None:
        save_dir = config.MODELS_DIR if config else 'results/models'

    print(f"\n[INFO] Preprocesando experimento: {experiment_name}")
    print(f"  Features originales: {X.shape[1]:,}")

    # Extraer los índices
    idx_train = splits['train_idx']
    idx_val = splits['val_idx']
    idx_test = splits['test_idx']

    # Función auxiliar para crear lotes (batches) y no saturar la RAM
    def get_batches(indices, batch_size=400, min_size=150):
        batches = []
        for i in range(0, len(indices), batch_size):
            batches.append(indices[i:i + batch_size])
        # Si el último lote es muy pequeño para el PCA, lo unimos al anterior
        if len(batches) > 1 and len(batches[-1]) < min_size:
            batches[-2] = np.concatenate([batches[-2], batches[-1]])
            batches.pop()
        return batches

    scaler = StandardScaler()
    
    # PASO A: Ajustar (fit) el Scaler en lotes
    print("Ajustando StandardScaler por lotes")
    for batch_idx in get_batches(idx_train):
        scaler.partial_fit(X[batch_idx])

    # PASO B: Ajustar (fit) el PCA en lotes
    if apply_pca:
        max_components = min(n_components, len(idx_train), X.shape[1])
        pca = IncrementalPCA(n_components=max_components)
        
        print("Ajustando IncrementalPCA por lotes")
        for batch_idx in get_batches(idx_train, min_size=max_components):
            batch_scaled = scaler.transform(X[batch_idx])
            pca.partial_fit(batch_scaled)
    else:
        pca = None

    # PASO C: Transformar los datos finales en lotes
    def transform_dataset(indices, dataset_name):
        print(f"Transformando conjunto {dataset_name}...")
        results = []
        for batch_idx in get_batches(indices):
            batch_scaled = scaler.transform(X[batch_idx])
            if apply_pca:
                batch_out = pca.transform(batch_scaled)
            else:
                batch_out = batch_scaled
            results.append(batch_out)
        return np.vstack(results)

    # Transformamos Train, Val y Test
    X_train_out = transform_dataset(idx_train, "Train")
    X_val_out = transform_dataset(idx_val, "Val")
    X_test_out = transform_dataset(idx_test, "Test")

    # Guardar los modelos preentrenados (Creamos un diccionario en lugar de un Pipeline)
    os.makedirs(save_dir, exist_ok=True)
    pkl_path = os.path.join(save_dir, f'preprocessor_{experiment_name}.pkl')
    preprocessor = {'scaler': scaler, 'pca': pca}
    joblib.dump(preprocessor, pkl_path)
    print(f"  Preprocesador guardado en: {pkl_path}")

    # Reportar
    print(f"  Features después de preprocesamiento: {X_train_out.shape[1]}")
    if apply_pca:
        var_explained = pca.explained_variance_ratio_.sum() * 100
        print(f"  PCA: {max_components} componentes ({var_explained:.1f}% varianza explicada)")

    return {
        'X_train': X_train_out, 'X_val': X_val_out, 'X_test': X_test_out,
        'y_train': labels[idx_train], 'y_val': labels[idx_val], 'y_test': labels[idx_test],
        'preprocessor': preprocessor,
    }



# 4. Pipeline completo


def preprocess_all(processed_dir=None):
    """
    Pipeline completo: carga datos, hace split, y preprocesa los 3 experimentos.
    
    Los 3 experimentos usan el MISMO split (mismos sujetos en train/val/test)
    para que la comparación sea justa.
    
    Experimentos:
      1. Solo FC    → StandardScaler + PCA(100) → 100 features
      2. Solo ALFF  → StandardScaler (sin PCA)  → 49 features
      3. FC + ALFF  → StandardScaler + PCA(100) → 100 features
    
    Returns:
        experiments: dict con keys 'fc', 'alff', 'combined'
                     cada uno contiene X_train, X_val, X_test, etc.
        splits: dict con los índices del split
    """

    # 1. Cargar datos
    data = load_processed_data(processed_dir)

    # 2. Hacer split (uno solo, compartido por los 3 experimentos)
    splits = make_splits(data['labels'])

    # 3. Preprocesar cada experimento
    experiments = {}

    # Experimento 1: Solo FC → necesita PCA (1.6M features)
    experiments['fc'] = preprocess_experiment(
        X=data['fc'],
        labels=data['labels'],
        splits=splits,
        experiment_name='fc',
        apply_pca=True
    )

    # Experimento 2: Solo ALFF → NO necesita PCA (49 features)
    experiments['alff'] = preprocess_experiment(
        X=data['alff'],
        labels=data['labels'],
        splits=splits,
        experiment_name='alff',
        apply_pca=False
    )

    # Experimento 3: FC + ALFF combinado → necesita PCA
    experiments['combined'] = preprocess_experiment(
        X=data['combined'],
        labels=data['labels'],
        splits=splits,
        experiment_name='combined',
        apply_pca=True
    )

    # 4. Resumen final
    for name, exp in experiments.items():
        print(f"\n  {name.upper()}:")
        print(f"    Train: {exp['X_train'].shape}")
        print(f"    Val:   {exp['X_val'].shape}")
        print(f"    Test:  {exp['X_test'].shape}")

    return experiments, splits


# Ejecucion directa (para testing)


if __name__ == '__main__':
    """
    Ejecutar para preprocesar los 3 experimentos:
        python src/preprocessing.py
    """
    try:
        experiments, splits = preprocess_all()

        # Guardar splits para reproducibilidad
        if config is not None:
            save_dir = config.PROCESSED_DIR
        else:
            save_dir = 'data/processed'

        np.save(os.path.join(save_dir, 'train_idx.npy'), splits['train_idx'])
        np.save(os.path.join(save_dir, 'val_idx.npy'), splits['val_idx'])
        np.save(os.path.join(save_dir, 'test_idx.npy'), splits['test_idx'])
        print(f"\nÍndices del split guardados en: {save_dir}")

        print("\n[INFO] Guardando matrices procesadas para no tener que recalcularlas...")
        for exp_name, exp_data in experiments.items():
            np.save(os.path.join(save_dir, f'X_train_{exp_name}.npy'), exp_data['X_train'])
            np.save(os.path.join(save_dir, f'X_val_{exp_name}.npy'), exp_data['X_val'])
            np.save(os.path.join(save_dir, f'X_test_{exp_name}.npy'), exp_data['X_test'])
            
            # Las labels son las mismas para todos, puedes guardarlas una sola vez si quieres
            np.save(os.path.join(save_dir, f'y_train_{exp_name}.npy'), exp_data['y_train'])
            np.save(os.path.join(save_dir, f'y_val_{exp_name}.npy'), exp_data['y_val'])
            np.save(os.path.join(save_dir, f'y_test_{exp_name}.npy'), exp_data['y_test'])
            
        print("[INFO] ¡Todos los datos procesados (PCA) han sido guardados con éxito!")

    except FileNotFoundError as e:
        print(f"\n[ERROR] Archivo no encontrado: {e}")
        print("Verifica que hayas corrido feature_extraction.py primero.")
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

