"""
preprocessing.py -- Preprocesamiento de features para ML clasico

Pasos:
  1. Cargar features y labels desde data/processed/
  2. Split 70/15/15 estratificado (mismo para los 3 experimentos)
  3. StandardScaler (fit en train, transform en val/test)
  4. PCA para FC y Combinado (ALFF con 49 features no lo necesita)
  5. Guardar preprocesadores como .pkl para reusar

Funciones:
  - load_processed_data()   : Carga los .npy de data/processed/
  - make_splits()           : Divide en train/val/test estratificado
  - preprocess_experiment() : Escala y aplica PCA a un experimento
  - preprocess_all()        : Pipeline completo para los 3 experimentos

Soporta dos modos via el parametro use_combat:
  - use_combat=False (default): usa fc_features.npy y alff_features.npy
  - use_combat=True           : usa fc_features_combat.npy y alff_features_combat.npy

Los archivos de salida llevan sufijo _combat cuando use_combat=True,
por lo que nunca se pisan los archivos sin armonizar.

Proyecto: Prediccion de depresion con IA -- Samsung Innovation Campus
Dataset: REST-meta-MDD Phase 1
"""

import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.feature_selection import SelectKBest, f_classif

# Importar configuracion
try:
    import config
except ImportError:
    print("[WARN] No se encontro config.py.")
    config = None

def load_processed_data(processed_dir=None, use_combat=False):
    """
    Carga los archivos .npy generados por feature_extraction.py
    (y opcionalmente armonizados por combat.py).

    El parametro use_combat decide que archivos leer:
      - False: fc_features.npy, alff_features.npy  (sin armonizar)
      - True:  fc_features_combat.npy, alff_features_combat.npy

    Args:
        processed_dir: ruta a data/processed/ (usa config si no se da)
        use_combat:    si True, carga las versiones armonizadas por ComBat

    Returns:
        data: dict con keys:
            'fc'    : array (2293, 1679028) -- features FC (mmap)
            'alff'  : array (2293, 49)      -- features ALFF
            'labels': array (2293,)          -- 1=MDD, 0=HC
            'ids'   : array (2293,)          -- subject IDs
    """
    if processed_dir is None:
        if config is not None:
            processed_dir = config.PROCESSED_DIR
        else:
            raise ValueError("Proporciona processed_dir o configura config.py")

    sufijo = '_combat' if use_combat else ''
    modo   = 'CON ComBat' if use_combat else 'SIN ComBat'

    print(f"[INFO] Cargando datos desde: {processed_dir}")
    print(f"[INFO] Modo: {modo}")

    fc_file   = os.path.join(processed_dir, f'fc_features{sufijo}.npy')
    alff_file = os.path.join(processed_dir, f'alff_features{sufijo}.npy')

    # Verificar que los archivos existen antes de intentar cargarlos
    for ruta in [fc_file, alff_file]:
        if not os.path.exists(ruta):
            raise FileNotFoundError(
                f"No se encontro: {ruta}\n"
                f"  Si use_combat=True, asegurate de haber corrido combat.py primero."
            )

    data = {
        'fc':     np.load(fc_file,   mmap_mode='r'),
        'alff':   np.load(alff_file),
        'labels': np.load(os.path.join(processed_dir, 'labels.npy')),
        'ids':    np.load(os.path.join(processed_dir, 'subject_ids.npy'),
                          allow_pickle=True),
    }

    print(f"[INFO] Datos cargados:")
    print(f"  FC:     {data['fc'].shape}  ({fc_file.split(os.sep)[-1]})")
    print(f"  ALFF:   {data['alff'].shape}  ({alff_file.split(os.sep)[-1]})")
    print(f"  Labels: {data['labels'].shape} "
          f"({data['labels'].sum()} MDD, {(data['labels'] == 0).sum()} HC)")

    return data


def make_splits(labels, test_size=None, val_size=None, random_state=None):
    """
    Genera los indices para dividir en train/val/test.

    Split por defecto: 70% train, 15% val, 15% test
    Estratificado: misma proporcion MDD/HC en cada set.

    Args:
        labels:       array de shape (n_sujetos,) con 0s y 1s
        test_size:    proporcion de test (default de config: 0.15)
        val_size:     proporcion de val (default de config: 0.15)
        random_state: semilla (default de config: 42)

    Returns:
        splits: dict con keys 'train_idx', 'val_idx', 'test_idx'
                cada uno es un array de indices
    """
    if test_size is None:
        test_size = config.TEST_SIZE if config else 0.15
    if val_size is None:
        val_size = config.VAL_SIZE if config else 0.15
    if random_state is None:
        random_state = config.RANDOM_STATE if config else 42

    indices = np.arange(len(labels))

    # Primer split: separar test (15%)
    idx_temp, idx_test = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    # Segundo split: separar val del restante 85%
    # val_relative = 0.15 / 0.85 = 0.176 para obtener 15% del total
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

    y_train = labels[idx_train]
    y_val   = labels[idx_val]
    y_test  = labels[idx_test]

    print(f"[INFO] Split completado:")
    print(f"  Train: {len(idx_train)} ({y_train.sum()} MDD, {(y_train == 0).sum()} HC)")
    print(f"  Val:   {len(idx_val)} ({y_val.sum()} MDD, {(y_val == 0).sum()} HC)")
    print(f"  Test:  {len(idx_test)} ({y_test.sum()} MDD, {(y_test == 0).sum()} HC)")

    return splits


def preprocess_experiment(X, labels, splits, experiment_name,
                          apply_pca=True, n_components=None, save_dir=None,
                          apply_anova=False, k_features=None,
                          apply_mrmr=False):
    """
    Aplica StandardScaler y opcionalmente IncrementalPCA a un experimento.

    Usa IncrementalPCA y procesamiento por lotes (batch_size=400) para
    no saturar la RAM con las matrices de 1.6M features.

    Args:
        X:               array (n_sujetos, n_features) — puede ser mmap
        labels:          array (n_sujetos,)
        splits:          dict de make_splits() con los indices
        experiment_name: nombre del experimento (ej. 'fc', 'fc_combat')
        apply_pca:       si aplicar PCA (False para ALFF con 49 features)
        n_components:    componentes PCA (default de config: 100)
        save_dir:        carpeta donde guardar el preprocesador .pkl

    Returns:
        result: dict con:
            'X_train', 'X_val', 'X_test': arrays preprocesados
            'y_train', 'y_val', 'y_test': arrays de labels
            'preprocessor': dict {'scaler': ..., 'pca': ...}
    """
    if n_components is None:
        n_components = config.PCA_COMPONENTS if config else 100
    if save_dir is None:
        save_dir = config.MODELS_DIR if config else 'results/models'
    if k_features is None:
        k_features = config.ANOVA_K_FEATURES if (config and hasattr(config, 'ANOVA_K_FEATURES')) else 100

    print(f"\n[INFO] Preprocesando experimento: {experiment_name}")
    print(f"  Features originales: {X.shape[1]:,}")

    idx_train = splits['train_idx']
    idx_val   = splits['val_idx']
    idx_test  = splits['test_idx']

    def get_batches(indices, batch_size=400, min_size=150):
        """Divide indices en lotes, fusionando el ultimo si es muy pequeno."""
        batches = []
        for i in range(0, len(indices), batch_size):
            batches.append(indices[i:i + batch_size])
        if len(batches) > 1 and len(batches[-1]) < min_size:
            batches[-2] = np.concatenate([batches[-2], batches[-1]])
            batches.pop()
        return batches

    # PASO A: Ajustar StandardScaler en lotes sobre train
    print("  Ajustando StandardScaler por lotes...")
    scaler = StandardScaler()
    for batch_idx in get_batches(idx_train):
        scaler.partial_fit(X[batch_idx])

    # PASO B: Ajustar IncrementalPCA en lotes sobre train (si aplica)
    if apply_pca:
        max_components = min(n_components, len(idx_train), X.shape[1])
        pca = IncrementalPCA(n_components=max_components)

        print("  Ajustando IncrementalPCA por lotes...")
        for batch_idx in get_batches(idx_train, min_size=max_components):
            batch_scaled = scaler.transform(X[batch_idx])
            pca.partial_fit(batch_scaled)
    else:
        pca = None

    if apply_anova:
        k = min(k_features, X.shape[1])
        print(f"  Calculando F-scores ANOVA de forma incremental (k={k})...")

        # Acumuladores float64 para evitar errores de precision numerica
        n_features = X.shape[1]
        n0 = 0; n1 = 0
        sum0    = np.zeros(n_features, dtype=np.float64)
        sum1    = np.zeros(n_features, dtype=np.float64)
        sum_sq0 = np.zeros(n_features, dtype=np.float64)
        sum_sq1 = np.zeros(n_features, dtype=np.float64)

        # Acumular estadisticas por clase en cada batch (nunca se apila todo)
        for batch_idx in get_batches(idx_train):
            batch_scaled = scaler.transform(X[batch_idx]).astype(np.float64)
            batch_labels = labels[batch_idx]
            mask0 = batch_labels == 0  # HC
            mask1 = batch_labels == 1  # MDD
            if mask0.any():
                n0      += mask0.sum()
                sum0    += batch_scaled[mask0].sum(axis=0)
                sum_sq0 += (batch_scaled[mask0] ** 2).sum(axis=0)
            if mask1.any():
                n1      += mask1.sum()
                sum1    += batch_scaled[mask1].sum(axis=0)
                sum_sq1 += (batch_scaled[mask1] ** 2).sum(axis=0)

        # Calcular medias y varianzas por clase
        mean0 = sum0 / n0
        mean1 = sum1 / n1
        var0  = np.maximum(sum_sq0 / n0 - mean0 ** 2, 0.0)
        var1  = np.maximum(sum_sq1 / n1 - mean1 ** 2, 0.0)

        # F-score de ANOVA de una via: varianza entre grupos / varianza dentro
        grand_mean    = (sum0 + sum1) / (n0 + n1)
        between_var   = (n0 * (mean0 - grand_mean) ** 2 +
                         n1 * (mean1 - grand_mean) ** 2)
        within_var    = np.maximum(n0 * var0 + n1 * var1, 1e-10)
        f_scores      = between_var / within_var

        # Indices de las top-k features por F-score descendente
        selector_indices = np.argsort(f_scores)[-k:]
        selector = {'indices': selector_indices, 'k': k}
        print(f"  ANOVA incremental completado: {k} features seleccionadas de {n_features:,}")

        del sum0, sum1, sum_sq0, sum_sq1, f_scores
    else:
        selector = None

    if apply_mrmr:
        from mrmr import mrmr_classif
        import pandas as pd

        k = min(k_features, X.shape[1])
        n_prefilter = min(5000, X.shape[1])
        print(f"  Calculando mRMR (prefiltrado ANOVA {n_prefilter} → mRMR k={k})...")

        # Paso 1: prefiltrado ANOVA incremental para reducir a 5000 features
        n_features = X.shape[1]
        n0 = 0; n1 = 0
        sum0    = np.zeros(n_features, dtype=np.float64)
        sum1    = np.zeros(n_features, dtype=np.float64)
        sum_sq0 = np.zeros(n_features, dtype=np.float64)
        sum_sq1 = np.zeros(n_features, dtype=np.float64)

        for batch_idx in get_batches(idx_train):
            batch_scaled = scaler.transform(X[batch_idx]).astype(np.float64)
            batch_labels = labels[batch_idx]
            mask0 = batch_labels == 0
            mask1 = batch_labels == 1
            if mask0.any():
                n0      += mask0.sum()
                sum0    += batch_scaled[mask0].sum(axis=0)
                sum_sq0 += (batch_scaled[mask0] ** 2).sum(axis=0)
            if mask1.any():
                n1      += mask1.sum()
                sum1    += batch_scaled[mask1].sum(axis=0)
                sum_sq1 += (batch_scaled[mask1] ** 2).sum(axis=0)

        mean0 = sum0 / n0
        mean1 = sum1 / n1
        var0  = np.maximum(sum_sq0 / n0 - mean0 ** 2, 0.0)
        var1  = np.maximum(sum_sq1 / n1 - mean1 ** 2, 0.0)
        grand_mean  = (sum0 + sum1) / (n0 + n1)
        between_var = (n0 * (mean0 - grand_mean) ** 2 +
                       n1 * (mean1 - grand_mean) ** 2)
        within_var  = np.maximum(n0 * var0 + n1 * var1, 1e-10)
        f_scores    = between_var / within_var
        prefilter_indices = np.argsort(f_scores)[-n_prefilter:]
        del sum0, sum1, sum_sq0, sum_sq1, f_scores, var0, var1

        print(f"  Prefiltrado ANOVA completado: {n_prefilter} features seleccionadas.")

        # Paso 2: materializar submatriz train prefiltrada (cabe en RAM)
        X_train_pre = np.vstack([
            scaler.transform(X[batch_idx])[:, prefilter_indices]
            for batch_idx in get_batches(idx_train)
        ])

        # Paso 3: mRMR sobre la submatriz prefiltrada
        print(f"  Ejecutando mRMR...")
        df_X = pd.DataFrame(X_train_pre.astype(np.float32))
        sr_y = pd.Series(labels[idx_train])
        selected_cols = mrmr_classif(X=df_X, y=sr_y, K=k, show_progress=True)
        mrmr_indices = prefilter_indices[np.array(selected_cols)]
        selector = {'indices': mrmr_indices, 'k': k}

        del X_train_pre, df_X, sr_y
        print(f"  mRMR completado: {k} features seleccionadas de {n_features:,}")
    elif not apply_anova:
        pass  # selector ya es None

    # PASO C: Transformar train, val y test en lotes
    def transform_dataset(indices, dataset_name):
        """Aplica scaler (y pca o anova si corresponde) por lotes."""
        print(f"  Transformando {dataset_name}...")
        results = []
        for batch_idx in get_batches(indices):
            batch_scaled = scaler.transform(X[batch_idx])
            if apply_pca:
                batch_out = pca.transform(batch_scaled)
            elif apply_anova or apply_mrmr:
                batch_out = batch_scaled[:, selector['indices']]
            else:
                batch_out = batch_scaled
            results.append(batch_out)
        return np.vstack(results)

    X_train_out = transform_dataset(idx_train, 'Train')
    X_val_out   = transform_dataset(idx_val,   'Val')
    X_test_out  = transform_dataset(idx_test,  'Test')

    # Guardar preprocesador como .pkl para reusar en inferencia
    os.makedirs(save_dir, exist_ok=True)
    pkl_path = os.path.join(save_dir, f'preprocessor_{experiment_name}.pkl')
    preprocessor = {'scaler': scaler, 'pca': pca, 'selector': selector}
    joblib.dump(preprocessor, pkl_path)
    print(f"  Preprocesador guardado: {pkl_path}")

    print(f"  Features tras preprocesamiento: {X_train_out.shape[1]}")
    if apply_pca:
        var_explained = pca.explained_variance_ratio_.sum() * 100
        print(f"  PCA: {max_components} componentes ({var_explained:.1f}% varianza explicada)")
    if apply_anova:
        print(f"  ANOVA: {k} features seleccionadas de {X.shape[1]:,}")
    if apply_mrmr:
        print(f"  mRMR: {k} features seleccionadas de {X.shape[1]:,}")

    return {
        'X_train': X_train_out,
        'X_val':   X_val_out,
        'X_test':  X_test_out,
        'y_train': labels[idx_train],
        'y_val':   labels[idx_val],
        'y_test':  labels[idx_test],
        'preprocessor': preprocessor,
    }


def preprocess_all(processed_dir=None, use_combat=False):
    """
    Pipeline completo: carga datos, hace split y preprocesa los 3 experimentos.

    Los 3 experimentos usan el MISMO split (mismos sujetos en train/val/test)
    para que la comparacion entre modelos sea justa.

    Experimentos:
      1. Solo FC    -> StandardScaler + PCA(100) -> 100 features
      2. Solo ALFF  -> StandardScaler (sin PCA)  -> 49 features
      3. Combinado  -> FC_PCA + ALFF_scaled concatenados -> 149 features
                       + StandardScaler final para igualar escalas

    Args:
        processed_dir: ruta a data/processed/ (usa config si no se da)
        use_combat:    si True, usa los features armonizados por ComBat

    Returns:
        experiments: dict con keys 'fc', 'alff', 'combined'
                     cada uno contiene X_train, X_val, X_test, y_train, etc.
        splits:      dict con los indices del split
    """
    if processed_dir is None:
        if config is not None:
            processed_dir = config.PROCESSED_DIR
        else:
            raise ValueError("Proporciona processed_dir o configura config.py")

    # Sufijo para nombres de archivos de salida
    sufijo = '_combat' if use_combat else ''

    # 1. Cargar datos (combat o sin combat segun el parametro)
    data = load_processed_data(processed_dir=processed_dir, use_combat=use_combat)

    # 2. Split unico compartido por los 3 experimentos
    splits = make_splits(data['labels'])

    experiments = {}

    # Experimento 1: Solo FC → PCA necesario (1.6M features)
    experiments['fc'] = preprocess_experiment(
        X=data['fc'],
        labels=data['labels'],
        splits=splits,
        experiment_name=f'fc{sufijo}',
        apply_pca=True
    )

    # Experimento 2: Solo ALFF → sin PCA (49 features)
    experiments['alff'] = preprocess_experiment(
        X=data['alff'],
        labels=data['labels'],
        splits=splits,
        experiment_name=f'alff{sufijo}',
        apply_pca=False
    )

    # Experimento 1b: Solo FC con ANOVA (seleccion de features reales)
    experiments['fc_anova'] = preprocess_experiment(
        X=data['fc'],
        labels=data['labels'],
        splits=splits,
        experiment_name=f'fc_anova{sufijo}',
        apply_pca=False,
        apply_anova=True
    )

    # Experimento 1c: Solo FC con mRMR (seleccion con minima redundancia)
    experiments['fc_mrmr'] = preprocess_experiment(
        X=data['fc'],
        labels=data['labels'],
        splits=splits,
        experiment_name=f'fc_mrmr{sufijo}',
        apply_pca=False,
        apply_mrmr=True,
        k_features=config.MRMR_K_FEATURES
    )

    print(f"\n[INFO] Construyendo experimento: combined{sufijo}")
    X_train_comb = np.concatenate(
        (experiments['fc']['X_train'], experiments['alff']['X_train']), axis=1
    )
    X_val_comb = np.concatenate(
        (experiments['fc']['X_val'], experiments['alff']['X_val']), axis=1
    )
    X_test_comb = np.concatenate(
        (experiments['fc']['X_test'], experiments['alff']['X_test']), axis=1
    )

    # Scaler final — fit solo en train
    final_scaler = StandardScaler()
    X_train_comb = final_scaler.fit_transform(X_train_comb)
    X_val_comb   = final_scaler.transform(X_val_comb)
    X_test_comb  = final_scaler.transform(X_test_comb)

    # Guardar el scaler del combinado
    save_dir = config.MODELS_DIR if config else 'results/models'
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(
        {'scaler': final_scaler, 'pca': None},
        os.path.join(save_dir, f'preprocessor_combined{sufijo}.pkl')
    )

    experiments['combined'] = {
        'X_train': X_train_comb,
        'X_val':   X_val_comb,
        'X_test':  X_test_comb,
        'y_train': experiments['fc']['y_train'],
        'y_val':   experiments['fc']['y_val'],
        'y_test':  experiments['fc']['y_test'],
    }

    print(f"\n[INFO] Construyendo experimento: combined_anova{sufijo}")
    X_train_comb_anova = np.concatenate(
        (experiments['fc_anova']['X_train'], experiments['alff']['X_train']), axis=1
    )
    X_val_comb_anova = np.concatenate(
        (experiments['fc_anova']['X_val'], experiments['alff']['X_val']), axis=1
    )
    X_test_comb_anova = np.concatenate(
        (experiments['fc_anova']['X_test'], experiments['alff']['X_test']), axis=1
    )

    final_scaler_anova = StandardScaler()
    X_train_comb_anova = final_scaler_anova.fit_transform(X_train_comb_anova)
    X_val_comb_anova   = final_scaler_anova.transform(X_val_comb_anova)
    X_test_comb_anova  = final_scaler_anova.transform(X_test_comb_anova)

    joblib.dump(
        {'scaler': final_scaler_anova, 'pca': None, 'selector': None},
        os.path.join(save_dir, f'preprocessor_combined_anova{sufijo}.pkl')
    )

    experiments['combined_anova'] = {
        'X_train': X_train_comb_anova,
        'X_val':   X_val_comb_anova,
        'X_test':  X_test_comb_anova,
        'y_train': experiments['fc']['y_train'],
        'y_val':   experiments['fc']['y_val'],
        'y_test':  experiments['fc']['y_test'],
    }

    print(f"\n[INFO] Construyendo experimento: combined_mrmr{sufijo}")
    X_train_comb_mrmr = np.concatenate(
        (experiments['fc_mrmr']['X_train'], experiments['alff']['X_train']), axis=1
    )
    X_val_comb_mrmr = np.concatenate(
        (experiments['fc_mrmr']['X_val'], experiments['alff']['X_val']), axis=1
    )
    X_test_comb_mrmr = np.concatenate(
        (experiments['fc_mrmr']['X_test'], experiments['alff']['X_test']), axis=1
    )

    final_scaler_mrmr = StandardScaler()
    X_train_comb_mrmr = final_scaler_mrmr.fit_transform(X_train_comb_mrmr)
    X_val_comb_mrmr   = final_scaler_mrmr.transform(X_val_comb_mrmr)
    X_test_comb_mrmr  = final_scaler_mrmr.transform(X_test_comb_mrmr)

    joblib.dump(
        {'scaler': final_scaler_mrmr, 'pca': None, 'selector': None},
        os.path.join(save_dir, f'preprocessor_combined_mrmr{sufijo}.pkl')
    )

    experiments['combined_mrmr'] = {
        'X_train': X_train_comb_mrmr,
        'X_val':   X_val_comb_mrmr,
        'X_test':  X_test_comb_mrmr,
        'y_train': experiments['fc']['y_train'],
        'y_val':   experiments['fc']['y_val'],
        'y_test':  experiments['fc']['y_test'],
    }

    # Resumen final
    print(f"\n[INFO] Resumen de experimentos ({('CON' if use_combat else 'SIN')} ComBat):")
    for name, exp in experiments.items():
        print(f"  {name.upper()}{sufijo}:")
        print(f"    Train: {exp['X_train'].shape}")
        print(f"    Val:   {exp['X_val'].shape}")
        print(f"    Test:  {exp['X_test'].shape}")

    return experiments, splits

if __name__ == '__main__':
    """
    Ejecutar para preprocesar los 3 experimentos:
        python src/preprocessing.py          # sin ComBat
        python src/preprocessing.py combat   # con ComBat
    """
    import sys

    use_combat = len(sys.argv) > 1 and sys.argv[1].lower() == 'combat'
    sufijo     = '_combat' if use_combat else ''

    if use_combat:
        print("[INFO] Con armonizacion ComBat")
    else:
        print("[INFO] Sin armonizacion ComBat (datos originales)")

    if config is not None:
        save_dir = config.PROCESSED_DIR
    else:
        save_dir = 'data/processed'

    # Verificacion por experimento individual
    # combined y combined_anova dependen de fc, alff y fc_anova respectivamente
    exp_nombres = ['fc', 'alff', 'fc_anova', 'fc_mrmr', 'combined', 'combined_anova', 'combined_mrmr']

    def cache_completo(exp_name):
        """Retorna True si los 6 archivos del experimento existen en disco."""
        archivos = [
            os.path.join(save_dir, f'X_train_{exp_name}{sufijo}.npy'),
            os.path.join(save_dir, f'X_val_{exp_name}{sufijo}.npy'),
            os.path.join(save_dir, f'X_test_{exp_name}{sufijo}.npy'),
            os.path.join(save_dir, f'y_train_{exp_name}{sufijo}.npy'),
            os.path.join(save_dir, f'y_val_{exp_name}{sufijo}.npy'),
            os.path.join(save_dir, f'y_test_{exp_name}{sufijo}.npy'),
        ]
        return all(os.path.exists(f) for f in archivos)

    def cargar_experimento(exp_name):
        """Carga los 6 archivos de un experimento desde disco."""
        return {
            'X_train': np.load(os.path.join(save_dir, f'X_train_{exp_name}{sufijo}.npy')),
            'X_val':   np.load(os.path.join(save_dir, f'X_val_{exp_name}{sufijo}.npy')),
            'X_test':  np.load(os.path.join(save_dir, f'X_test_{exp_name}{sufijo}.npy')),
            'y_train': np.load(os.path.join(save_dir, f'y_train_{exp_name}{sufijo}.npy')),
            'y_val':   np.load(os.path.join(save_dir, f'y_val_{exp_name}{sufijo}.npy')),
            'y_test':  np.load(os.path.join(save_dir, f'y_test_{exp_name}{sufijo}.npy')),
        }

    def guardar_experimento(exp_name, exp_data):
        """Guarda los 6 archivos de un experimento en disco."""
        np.save(os.path.join(save_dir, f'X_train_{exp_name}{sufijo}.npy'), exp_data['X_train'])
        np.save(os.path.join(save_dir, f'X_val_{exp_name}{sufijo}.npy'),   exp_data['X_val'])
        np.save(os.path.join(save_dir, f'X_test_{exp_name}{sufijo}.npy'),  exp_data['X_test'])
        np.save(os.path.join(save_dir, f'y_train_{exp_name}{sufijo}.npy'), exp_data['y_train'])
        np.save(os.path.join(save_dir, f'y_val_{exp_name}{sufijo}.npy'),   exp_data['y_val'])
        np.save(os.path.join(save_dir, f'y_test_{exp_name}{sufijo}.npy'),  exp_data['y_test'])

    try:
        # Cargar o generar splits
        splits_path = os.path.join(save_dir, 'train_idx.npy')
        if os.path.exists(splits_path):
            splits = {
                'train_idx': np.load(os.path.join(save_dir, 'train_idx.npy')),
                'val_idx':   np.load(os.path.join(save_dir, 'val_idx.npy')),
                'test_idx':  np.load(os.path.join(save_dir, 'test_idx.npy')),
            }
        else:
            data = load_processed_data(use_combat=use_combat)
            splits = make_splits(data['labels'])
            np.save(os.path.join(save_dir, 'train_idx.npy'), splits['train_idx'])
            np.save(os.path.join(save_dir, 'val_idx.npy'),   splits['val_idx'])
            np.save(os.path.join(save_dir, 'test_idx.npy'),  splits['test_idx'])
            print(f"[INFO] Indices del split guardados en: {save_dir}")

        experiments = {}
        necesita_datos = False

        # Determinar cuales experimentos necesitan procesarse
        for exp_name in exp_nombres:
            if cache_completo(exp_name):
                print(f"[INFO] Cache encontrado: {exp_name}{sufijo} — cargando desde disco.")
                experiments[exp_name] = cargar_experimento(exp_name)
            else:
                print(f"[INFO] Cache faltante: {exp_name}{sufijo} — se procesara.")
                necesita_datos = True

        # Si hay algun experimento faltante, cargar los datos raw
        if necesita_datos:
            data = load_processed_data(use_combat=use_combat)

            # fc
            if 'fc' not in experiments:
                experiments['fc'] = preprocess_experiment(
                    X=data['fc'], labels=data['labels'], splits=splits,
                    experiment_name=f'fc{sufijo}', apply_pca=True
                )
                guardar_experimento(f'fc{sufijo}', experiments['fc'])

            # alff
            if 'alff' not in experiments:
                experiments['alff'] = preprocess_experiment(
                    X=data['alff'], labels=data['labels'], splits=splits,
                    experiment_name=f'alff{sufijo}', apply_pca=False
                )
                guardar_experimento(f'alff{sufijo}', experiments['alff'])

            # fc_anova
            if 'fc_anova' not in experiments:
                experiments['fc_anova'] = preprocess_experiment(
                    X=data['fc'], labels=data['labels'], splits=splits,
                    experiment_name=f'fc_anova{sufijo}', apply_pca=False, apply_anova=True
                )
                guardar_experimento(f'fc_anova{sufijo}', experiments['fc_anova'])

            # fc_mrmr
            if 'fc_mrmr' not in experiments:
                experiments['fc_mrmr'] = preprocess_experiment(
                    X=data['fc'], labels=data['labels'], splits=splits,
                    experiment_name=f'fc_mrmr{sufijo}', apply_pca=False, apply_mrmr=True
                )
                guardar_experimento(f'fc_mrmr{sufijo}', experiments['fc_mrmr'])

            # combined — depende de fc y alff
            if 'combined' not in experiments:
                print(f"\n[INFO] Construyendo experimento: combined{sufijo}")
                X_train_comb = np.concatenate(
                    (experiments['fc']['X_train'], experiments['alff']['X_train']), axis=1
                )
                X_val_comb = np.concatenate(
                    (experiments['fc']['X_val'], experiments['alff']['X_val']), axis=1
                )
                X_test_comb = np.concatenate(
                    (experiments['fc']['X_test'], experiments['alff']['X_test']), axis=1
                )
                final_scaler = StandardScaler()
                X_train_comb = final_scaler.fit_transform(X_train_comb)
                X_val_comb   = final_scaler.transform(X_val_comb)
                X_test_comb  = final_scaler.transform(X_test_comb)

                save_models = config.MODELS_DIR if config else 'results/models'
                os.makedirs(save_models, exist_ok=True)
                joblib.dump(
                    {'scaler': final_scaler, 'pca': None},
                    os.path.join(save_models, f'preprocessor_combined{sufijo}.pkl')
                )

                experiments['combined'] = {
                    'X_train': X_train_comb,
                    'X_val':   X_val_comb,
                    'X_test':  X_test_comb,
                    'y_train': experiments['fc']['y_train'],
                    'y_val':   experiments['fc']['y_val'],
                    'y_test':  experiments['fc']['y_test'],
                }
                guardar_experimento(f'combined{sufijo}', experiments['combined'])

            # combined_anova — depende de fc_anova y alff
            if 'combined_anova' not in experiments:
                print(f"\n[INFO] Construyendo experimento: combined_anova{sufijo}")
                X_train_comb_anova = np.concatenate(
                    (experiments['fc_anova']['X_train'], experiments['alff']['X_train']), axis=1
                )
                X_val_comb_anova = np.concatenate(
                    (experiments['fc_anova']['X_val'], experiments['alff']['X_val']), axis=1
                )
                X_test_comb_anova = np.concatenate(
                    (experiments['fc_anova']['X_test'], experiments['alff']['X_test']), axis=1
                )
                final_scaler_anova = StandardScaler()
                X_train_comb_anova = final_scaler_anova.fit_transform(X_train_comb_anova)
                X_val_comb_anova   = final_scaler_anova.transform(X_val_comb_anova)
                X_test_comb_anova  = final_scaler_anova.transform(X_test_comb_anova)

                save_models = config.MODELS_DIR if config else 'results/models'
                joblib.dump(
                    {'scaler': final_scaler_anova, 'pca': None, 'selector': None},
                    os.path.join(save_models, f'preprocessor_combined_anova{sufijo}.pkl')
                )

                experiments['combined_anova'] = {
                    'X_train': X_train_comb_anova,
                    'X_val':   X_val_comb_anova,
                    'X_test':  X_test_comb_anova,
                    'y_train': experiments['fc']['y_train'],
                    'y_val':   experiments['fc']['y_val'],
                    'y_test':  experiments['fc']['y_test'],
                }
                guardar_experimento(f'combined_anova{sufijo}', experiments['combined_anova'])

            # combined_mrmr — depende de fc_mrmr y alff
            if 'combined_mrmr' not in experiments:
                print(f"\n[INFO] Construyendo experimento: combined_mrmr{sufijo}")
                X_train_comb_mrmr = np.concatenate(
                    (experiments['fc_mrmr']['X_train'], experiments['alff']['X_train']), axis=1
                )
                X_val_comb_mrmr = np.concatenate(
                    (experiments['fc_mrmr']['X_val'], experiments['alff']['X_val']), axis=1
                )
                X_test_comb_mrmr = np.concatenate(
                    (experiments['fc_mrmr']['X_test'], experiments['alff']['X_test']), axis=1
                )
                final_scaler_mrmr = StandardScaler()
                X_train_comb_mrmr = final_scaler_mrmr.fit_transform(X_train_comb_mrmr)
                X_val_comb_mrmr   = final_scaler_mrmr.transform(X_val_comb_mrmr)
                X_test_comb_mrmr  = final_scaler_mrmr.transform(X_test_comb_mrmr)

                save_models = config.MODELS_DIR if config else 'results/models'
                joblib.dump(
                    {'scaler': final_scaler_mrmr, 'pca': None, 'selector': None},
                    os.path.join(save_models, f'preprocessor_combined_mrmr{sufijo}.pkl')
                )

                experiments['combined_mrmr'] = {
                    'X_train': X_train_comb_mrmr,
                    'X_val':   X_val_comb_mrmr,
                    'X_test':  X_test_comb_mrmr,
                    'y_train': experiments['fc']['y_train'],
                    'y_val':   experiments['fc']['y_val'],
                    'y_test':  experiments['fc']['y_test'],
                }
                guardar_experimento(f'combined_mrmr{sufijo}', experiments['combined_mrmr'])

            print("[INFO] Todos los datos procesados guardados con exito.")
            print(f"       Sufijo: '{sufijo}' (vacio = sin ComBat)")

        # Resumen final
        print(f"\n[INFO] Resumen de datos ({('CON' if use_combat else 'SIN')} ComBat):")
        for name in exp_nombres:
            exp = experiments[name]
            print(f"  {name.upper()}{sufijo}:")
            print(f"    Train: {exp['X_train'].shape}")
            print(f"    Val:   {exp['X_val'].shape}")
            print(f"    Test:  {exp['X_test'].shape}")

    except FileNotFoundError as e:
        print(f"\n[ERROR] Archivo no encontrado: {e}")
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()