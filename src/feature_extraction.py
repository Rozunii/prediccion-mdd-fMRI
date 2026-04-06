"""
feature_extraction.py — Extracción de features para clasificación

Este módulo toma los datos del data_loader.py y los transforma
en vectores de features.

Genera 3 tipos de features:
  1. FC (Functional Connectivity): Matriz de correlación entre ROIs aplanada
  2. ALFF: Actividad cerebral promediada por regiones del atlas AAL
  3. Combinado: FC + ALFF concatenados

Funciones principales:
  - exclude_incompatible_subjects()  : Filtra sujetos con ROIs diferentes (ej: S19)
  - build_fc_matrix()                : Construye matriz de conectividad de un sujeto
  - extract_fc_features()            : Extrae features FC de todos los sujetos
  - extract_alff_features()          : Extrae features ALFF de todos los sujetos
  - extract_combined_features()      : Concatena FC + ALFF
  - get_labels()                     : Obtiene vector de etiquetas alineado

Proyecto: Predicción de depresión con IA — Samsung Innovation Campus
Dataset: REST-meta-MDD Phase 1
"""


import numpy as np
import warnings
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker

# Importar config 
try:
    import config
except ImportError:
    print('[WARN] No se encontro config.py')
    config = None

# Filtrar sujetos con distintos ROIS
def exclude_incompatible_subjects(roi_signals, alff_volumes, metadata, common_ids, expected_rois=1833):
    """
    Filtra sujetos cuyo número de ROIs no coincide con el esperado.
    
    El sitio S19 tiene 1568 ROIs en vez de 1833. Para que todas las
    matrices FC tengan el mismo tamaño, excluimos esos sujetos.
    No los borramos del disco — solo los sacamos del análisis.
    
    Args:
        roi_signals: dict {subject_id: array (timepoints, n_rois)}
        alff_volumes: dict {subject_id: array (61, 73, 61)}
        metadata: DataFrame con columna 'ID' y 'label'
        common_ids: lista de IDs disponibles
        expected_rois: número de ROIs que deben tener (1833)
    
    Returns:
        roi_signals_clean: dict solo con sujetos compatibles
        alff_volumes_clean: dict solo con sujetos compatibles
        metadata_clean: DataFrame filtrado
        valid_ids: lista de IDs que pasaron el filtro
        excluded_ids: lista de IDs excluidos (para reportar)
    """

    valid_ids = []
    excluded_ids = []

    for sid in common_ids:
        n_rois = roi_signals[sid].shape[1]
        if n_rois == expected_rois:
            valid_ids.append(sid)
        else:
            excluded_ids.append(sid)

    # Filtrar los diccionarios
    roi_signals_clean = {sid: roi_signals[sid] for sid in valid_ids}
    alff_volumes_clean = {sid: alff_volumes[sid] for sid in valid_ids}

    # Filtrar metadata
    metadata_clean = metadata[metadata['ID'].isin(valid_ids)].copy()
    metadata_clean = metadata_clean.set_index('ID').loc[valid_ids].reset_index()

    # Reportar lo excluido
    if excluded_ids:
        sites_excluded = set(sid.split('-')[0] for sid in excluded_ids)
        print(f"[INFO] Sujetos exluidos: {len(excluded_ids)}" 
               f"Sitios: {', '.join(sorted(sites_excluded))}\n")
        
    print(f"[INFO] Sujetos válidos: {len(valid_ids)} "
          f"({metadata_clean['label'].sum()} MDD, "
          f"{(metadata_clean['label'] == 0).sum()} HC)")
    
    return roi_signals_clean, alff_volumes_clean, metadata_clean, valid_ids, excluded_ids
        
# Features
def build_fc_matrix(roi_timeseries):
    """
    Construye la matriz de conectividad funcional de UN sujeto.
    
    Proceso:
      1. Correlación de Pearson entre todos los pares de ROIs
         → Si tenemos 1833 ROIs, esto genera una matriz de 1833×1833
         → Cada celda [i,j] dice qué tan correlacionada está la
           actividad de la región i con la región j
      2. Fisher z-transform (arctanh)
         → Normaliza los valores de correlación para que se puedan
           comparar estadísticamente entre sujetos
      3. Diagonal a cero
         → La correlación de una región consigo misma siempre es 1,
           no aporta información, así que la ponemos en 0
    
    Args:
        roi_timeseries: array de shape (n_timepoints, n_rois)
                        Cada columna es la serie temporal de una región
    
    Returns:
        fc_matrix: array de shape (n_rois, n_rois) — matriz simétrica
    """   

    # Correlacion de Pearson con las variables
    fc_matrix = np.corrcoef(roi_timeseries.T)

    # Manejar varianza = 0 o NaN
    fc_matrix = np.nan_to_num(fc_matrix, nan=0.0)

    # Recortamos valores para evitar infinitos
    np.clip(fc_matrix, -0.999, 0.999, out=fc_matrix)

    # Aplicar Fisher
    fc_matrix = np.arctanh(fc_matrix)

    # Poner la diagonal en 0
    np.fill_diagonal(fc_matrix, 0)

    return fc_matrix

def fc_matrix_to_vector(fc_matrix):
    """
    Extrae el triángulo superior de la matriz FC como un vector.
    
    La matriz FC es simétrica (la correlación de A con B es igual
    a la de B con A), así que solo necesitamos la mitad.
    El triángulo superior (sin la diagonal) contiene toda la info.
    
    Para 1833 ROIs: 1833 × 1832 / 2 = 1,679,028 features
    
    Args:
        fc_matrix: array de shape (n_rois, n_rois)
    
    Returns:
        vector: array de shape (n_features,)
    """

    # Obtener los indices del triagulo superior
    indices = np.triu_indices_from(fc_matrix, k=1)
    return fc_matrix[indices]

def extract_fc_features(roi_signals, subject_id):
    """
    Extrae features de conectividad funcional para todos los sujetos.
    
    Para cada sujeto:
      1. Toma sus series temporales (200 timepoints × 1833 ROIs)
      2. Calcula la matriz de correlación (1833 × 1833)
      3. Aplica Fisher z-transform
      4. Aplana el triángulo superior en un vector
    
    Args:
        roi_signals: dict {subject_id: array (timepoints, n_rois)}
        subject_ids: lista de IDs en el orden deseado
    
    Returns:
        X_fc: array de shape (n_sujetos, n_fc_features)
              donde n_fc_features = n_rois * (n_rois - 1) / 2
    """

    n_subjects = len(subject_id)

    # Calcular numero de features
    first_fc = build_fc_matrix(roi_signals[subject_id[0]])
    first_vec = fc_matrix_to_vector(first_fc)
    n_features = len(first_vec)

    print(f"[INFO] Features por sujeto: {n_features:,} "
          f"(triángulo superior de {first_fc.shape[0]}×{first_fc.shape[1]})\n")
    
    # Pre-alocar la matriz
    X_fc = np.zeros((n_subjects, n_features), dtype=np.float32)

    # Llenar fila por fila
    for i, sid in enumerate(subject_id):
        fc = build_fc_matrix(roi_signals[sid])
        X_fc[i] = fc_matrix_to_vector(fc)

        # Progreso
        if (i + 1) % 200 == 0:
            print(f" ... {i + 1}/{n_subjects} procesados")

    print(f"\n[INFO] Matriz completada: {X_fc.shape}")

    return X_fc

# Features de ALFF
def _load_atlas():
    """
    Carga el atlas Harvard-Oxford cortical.
    
    Divide el cerebro en 48 regiones corticales conocidas
    (frontal pole, insular cortex, superior temporal gyrus, etc.).
    Ya viene incluido con nilearn, no requiere descarga externa.
    
    Returns:
        masker: objeto NiftiLabelsMasker listo para transformar volúmenes
        region_names: lista con los nombres de las regiones
    """

    print("\n[INFO] Cargando atlas Harvard-Oxford...")
    atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')

    # Crear el masker (promedia voxels)
    masker = NiftiLabelsMasker(
        labels_img=atlas.maps,
        standardize=False,
        resampling_target='data',
        strategy='mean'
    )

    region_names = atlas.labels
    
    return masker, region_names

def extract_alff_features(alff_volumes, subject_ids, alff_dir=None):
    """
    Extrae features ALFF promediados por regiones del atlas AAL.
    
    Para cada sujeto:
      1. Toma su volumen ALFF (61×73×61)
      2. Usa el atlas AAL para identificar las 116 regiones
      3. Promedia todos los vóxeles dentro de cada región
      4. Resultado: un vector de 116 valores (uno por región)
    
    Args:
        alff_volumes: dict {subject_id: array} — no se usa directamente,
                      pero sirve para verificar qué sujetos tenemos
        subject_ids: lista de IDs en el orden deseado
        alff_dir: ruta a la carpeta con archivos .nii (usa config si no se da)
    
    Returns:
        X_alff: array de shape (n_sujetos, 116)
        region_names: lista con nombres de las regiones
    """

    import os

    if alff_dir == None:
        if config is not None:
            alff_dir = config.ALFF_DIR
        else:
            raise ValueError('Proporciona alff_dir o configura config.py')
        
    n_subjects = len(subject_ids)

    # Cargar atlas AAL
    masker, region_names = _load_atlas()

    # Entrenar atlas con un archivo de ejemplo
    example_file = os.path.join(alff_dir, f"ALFFMap_{subject_ids[0]}.nii.gz")
    masker.fit(example_file)

    n_regions = len(region_names)
    print(f"\n[INFO] Atlas AAL: {n_regions} regiones")

    # Pre-alocar
    X_alff = np.zeros((n_subjects, n_regions), dtype=np.float32)

    # Extraer features
    for i, sid in enumerate(subject_ids):
        nii_path = os.path.join(alff_dir, f"ALFFMap_{sid}.nii.gz")

        try:
            features = masker.transform(nii_path).ravel()
            X_alff[i, :len(features)] = features
        
        except Exception as e:
            warnings.warn(f"Error en ALFF {sid}: {e}")

        # Progreso
        if (i + 1) % 200 == 0:
            print(f" ... {i + 1}/{n_subjects} procesados")
        
    print(f"\n[INFO] Matriz completa: {X_alff.shape}")

    return X_alff, region_names

# Combinar features
def extract_combined_features(X_fc, X_alff):
    """
    Concatena los features FC y ALFF en un solo vector por sujeto.
    
    Simplemente pone los features FC y ALFF uno al lado del otro.
    Si FC tiene 1,679,028 features y ALFF tiene 116, el resultado
    tiene 1,679,144 features por sujeto.
    
    Args:
        X_fc: array de shape (n_sujetos, n_fc_features)
        X_alff: array de shape (n_sujetos, n_alff_features)
    
    Returns:
        X_combined: array de shape (n_sujetos, n_fc + n_alff)
    """

    if X_fc.shape[0] != X_alff.shape[0]:
        raise ValueError(
            f"FC tiene {X_fc.shape[0]} sujetos pero ALFF tiene {X_alff.shape[0]}. "
            f"Deben ser iguales."
        )

    X_combined = np.concatenate([X_fc, X_alff], axis=1)

    print(f"\n[INFO] Features combinados: {X_combined.shape} "
          f"(FC: {X_fc.shape[1]:,} + ALFF: {X_alff.shape[1]})\n")

    return X_combined

# Obtener labels
def get_labels(metadata, subject_ids):
    """
    Obtiene el vector de etiquetas alineado con los features.
    
    Cuando extraemos features, los sujetos quedan en cierto orden
    (definido por subject_ids). Esta función crea un array de 0s y 1s
    en ese mismo orden, para que X[i] corresponda con y[i].
    
    Args:
        metadata: DataFrame con columnas 'ID' y 'label'
        subject_ids: lista de IDs en el mismo orden que los features
    
    Returns:
        y: array de shape (n_sujetos,) con valores 0 (HC) y 1 (MDD)
    """

    # Crear un diccionario ID → label para busqueda.
    id_to_label = dict(zip(metadata['ID'], metadata['label']))

    # Construir el array de labels
    y = np.array([id_to_label[sid] for sid in subject_ids])

    mdd_count = y.sum()
    hc_count = (y == 0).sum()
    print(f"[INFO] Labels: {len(y)} total ({mdd_count} MDD, {hc_count} HC)\n")

    return y

# Ejecucion testing
if __name__ == '__main__':
    """
    Ejecutar para extraer todos los features y guardarlos en disco:
        python src/feature_extraction.py
    """
    import os
    from data_loader import load_all_data

    try:
        # 1. Cargar datos crudos
        metadata, roi_signals, alff_volumes, common_ids = load_all_data()

        # 2. Excluir sujetos incompatibles
        roi_clean, alff_clean, meta_clean, valid_ids, excluded = \
            exclude_incompatible_subjects(
                roi_signals, alff_volumes, metadata, common_ids
            )

        # 3. Extraer features FC
        X_fc = extract_fc_features(roi_clean, valid_ids)

        # 4. Extraer features ALFF
        X_alff, region_names = extract_alff_features(alff_clean, valid_ids)

        # 5. Combinar
        X_combined = extract_combined_features(X_fc, X_alff)

        # 6. Labels
        y = get_labels(meta_clean, valid_ids)

        # 7. Guardar en disco
        if config is not None:
            save_dir = config.PROCESSED_DIR
        else:
            save_dir = "data/processed"

        os.makedirs(save_dir, exist_ok=True)

        np.save(os.path.join(save_dir, 'fc_features.npy'), X_fc)
        np.save(os.path.join(save_dir, 'alff_features.npy'), X_alff)
        np.save(os.path.join(save_dir, 'combined_features.npy'), X_combined)
        np.save(os.path.join(save_dir, 'labels.npy'), y)
        np.save(os.path.join(save_dir, 'subject_ids.npy'), np.array(valid_ids))

        print(f"\nArchivos guardados en: {save_dir}")
        print(f"  fc_features.npy:       {X_fc.shape}")
        print(f"  alff_features.npy:     {X_alff.shape}")
        print(f"  combined_features.npy: {X_combined.shape}")
        print(f"  labels.npy:            {y.shape}")
        print(f"  subject_ids.npy:       {len(valid_ids)} IDs")

    except FileNotFoundError as e:
        print(f"\n[ERROR] Archivo no encontrado: {e}")
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()