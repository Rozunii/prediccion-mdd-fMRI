"""
data_loader.py — Carga y preprocesamiento de datos REST-meta-MDD

Funciones principales:
  - load_metadata()      : Carga el Excel con info demográfica/clínica
  - load_roi_signals()   : Carga archivos .mat con series temporales por ROI
  - build_fc_matrix()    : Construye matriz de conectividad funcional
  - prepare_dataset()    : Pipeline completo → X, y listos para entrenar
  - get_splits()         : Divide en train/val/test estratificado

Proyecto: Predicción de depresión con IA — Samsung Innovation Campus
Dataset: REST-meta-MDD Phase 1
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio
import nibabel as nib 
import warnings

try: 
    import config 
except ImportError:
    print("[WARN] No se encontró config.py. Usando valores por defecto.")
    config = None


# Cargas datos
def parse_subject_id(subject_id):
    """
    Parsea un ID de sujeto para extraer sitio, grupo y número.
    
    Formato: S{sitio}-{grupo}-{número}
    Ejemplo: 'S1-1-0001' → site='S1', group=1 (MDD), number='0001'
             'S1-2-0001' → site='S1', group=2 (HC),  number='0001'
    
    Returns:
        dict con keys: 'site', 'group', 'number', 'label'
              label: 1 = MDD (depresión), 0 = HC (control sano)
    """

    parts = str(subject_id).split('-')
    if len(parts) != 3:
        raise ValueError(f'ID con formato irregular: {subject_id}')
    
    site = parts[0] # S1, S2, S3...
    group = int(parts[1]) # 1 = MDD, 2 = HC
    number = parts[2] # 0001, 0002, 0003...

    # Convertir un grupo a label binario (Con depresion y sin depresion)
    label = 1 if group == 1 else 0

    return {
        'site': site,
        'group': group,
        'number': number,
        'label': label
    }

def load_metadata(metadata_path=None):
    """
    Carga el archivo Excel de data con ambas hojas (MDD y Controls).
    
    El Excel tiene dos hojas:
      - 'MDD': 1276 sujetos, 33 columnas (ID, Sex, Age, Education, HAMD, etc.)
      - 'Controls': 1104 sujetos, 4 columnas (ID, Sex, Age, Education)
    
    Returns:
        pd.DataFrame con columnas: ID, Sex, Age, Education, label, site
        + columnas clínicas para MDD (HAMD, etc.) con NaN para controles
    """
    if metadata_path is None:
        if config is not None:
            metadata_path = config.METADATA_FILE
        else:
            raise ValueError('Proporciona metadata_path o actualiza config.py')
    
    print(f"[INFO] Cargando metadata desde: {metadata_path}")
    
    # Cargar datos MDD
    df_mdd = pd.read_excel(metadata_path, sheet_name='MDD')
    df_mdd['label'] = 1 # MDD = 1

    # Limpiar valores 
    df_mdd.replace(-9999, np.nan, inplace=True)
    df_mdd.replace('[]', np.nan, inplace=True)

    # Cargar datos HC
    df_hc = pd.read_excel(metadata_path, sheet_name='Controls')
    df_hc['label'] = 0 # HC = 0

    # Unificar 
    # HC solo tiene cuatro columnas lo demas queda como NaN
    df = pd.concat([df_mdd, df_hc], ignore_index=True, sort=False)

    # Extraer ID con la variable site
    df['site'] = df['ID'].apply(lambda x: str(x).split('-')[0])

    # Renombrar variables de los metadatos para mejor manipulacion
    rename_map = {
        'Education (years)': 'Education',
        'Illness duration (months)': 'Illness_duration',
        'If first episode?': 'First_episode',
        'On medication?': 'On_medication'
    }
    df.rename(columns=rename_map, inplace=True)

    print(f'\n[INFO] Metadatos: {len(df)} sujetos\n Con depresion: {df['label'].sum()}\n Sanos: {(df['label']==0).sum()}\n')

    return df

def load_single_roi(mat_path):
    """
    Carga un archivo .mat de ROISignals para un sujeto.
    
    Los archivos .mat contienen una variable (generalmente 'ROISignals'
    o similar) con shape (n_timepoints, n_rois) donde n_rois = 116 (atlas AAL).
    
    Returns:
        np.ndarray de shape (n_timepoints, 116)
    """

    mat = sio.loadmat(mat_path)

    # Buscar la variable de datos
    data_keys = [k for k in mat.keys() if not k.startswith('__')]

    if len(data_keys) == 0:
        raise ValueError(f'No se encontraron datos en {mat_path}')
    
    
    data = None
    for key in data_keys: 
        arr = mat[key]
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            data = arr
            break
    
    if data is None:
        raise ValueError(f'No se encontraron matriz 2D en {mat_path}')
    
    return data 

def _extract_id_from_filename(filename):
    """
    Extrae el subject ID del nombre de un archivo.

    Ejemplos:
        'ROISignals_S1-1-0001.mat' → 'S1-1-0001'
        'S1-1-0001.mat'            → 'S1-1-0001'
        'ALFFMap_S1-1-0001.nii.gz' → 'S1-1-0001'
    """
    
    # Quitar extensión(es)
    name = filename
    for ext in ['.nii.gz', '.nii', '.mat']:
        if name.endswith(ext):
            name = name[:-len(ext)]
            break
    
    # Quitar prefijo si existe
    for p in ['ROISignals_', 'ALFFMap_']:
        if name.startswith(p):
            name = name[len(p):]
            break
        
    if name.startswith('S') and name.count('-') == 2:
        return name
    
    return None



def load_roi_signals(roi_dir = None, metadata_df = None):
    """
    Carga todos los archivos .mat de ROISignals y los asocia con su metadata.
    
    Args:
        roi_dir: Ruta a la carpeta con archivos .mat
        metadata_df: DataFrame de metadata (de load_metadata())
        n_rois: Número esperado de ROIs (116 para atlas AAL)
    
    Returns:
        dict con key=subject_id, value=np.ndarray (n_timepoints, n_rois)
        list de IDs que se cargaron exitosamente
    """    

    if roi_dir == None:
        if config is not None:
            roi_dir = config.ROI_DIR
        else:
            raise ValueError('Debe proporcionar un roi_dir o tener config.py')
        
    print(f"[INFO] Cargando ROISignals desde: {roi_dir}")
    
    # Listar archivos .mat disponibles
    mat_files = [f for f in os.listdir(roi_dir) if f.endswith('.mat')]
    print(f'[INFO] Archivos .mat encontrados: {len(mat_files)}')

    # Cargar señales
    signals = {}
    loaded_ids = []
    failed_ids = []
    skipped = []

    for mat_file in sorted(mat_files):
        subject_id = _extract_id_from_filename(mat_file)
    
        if subject_id is None:
            skipped.append(mat_file)
            continue

        # Solo cargar datos que esten en los metadatos
        if metadata_df is not None and subject_id not in metadata_df['ID'].values:
            continue

        try:
            roi_data = load_single_roi(os.path.join(roi_dir, mat_file))
            signals[subject_id] = roi_data
            loaded_ids.append(subject_id)

        except Exception as e:
            failed_ids.append(subject_id)
            if len(failed_ids) <= 5:
                print(f"[WARN] Error cargando {subject_id}: {e}")

        # Progreso cada 500 archivos
        total_processed = len(loaded_ids) + len(failed_ids) + len(skipped)
        if total_processed % 500 == 0:
            print(f" ... {total_processed}/{len(mat_files)} procesados")

    if skipped:
        print(f"[WARN] {len(skipped)} archivos sin ID reconocible")
    if failed_ids:
        print(f"[WARN] {len(failed_ids)} archivos fallidos")

    print(f"[INFO] ROISignals cargados: {len(loaded_ids)} sujetos\n")

    return signals, loaded_ids, failed_ids

def load_single_alff(nii_patch):
    """
    Carga un archivo .nii/.nii.gz de ALFF para un sujeto.

    Cada archivo ALFF es un volumen 3D de shape (61, 73, 61)
    que representa la amplitud de fluctuaciones de baja frecuencia
    en cada vóxel del cerebro.

    Returns:
        np.ndarray de shape (61, 73, 61)
    """

    img = nib.load(str(nii_patch))
    data = img.get_fdata()

    return data

def load_alff_volumes(alff_dir = None,  metadata_df = None):
    """
    Carga todos los archivos .nii de ALFF.

    Args:
        alff_dir: Ruta a la carpeta con archivos .nii/.nii.gz
        metadata_df: DataFrame de metadata (para filtrar solo sujetos válidos)

    Returns:
        volumes: dict {subject_id: np.ndarray (61, 73, 61)}
        loaded_ids: list de IDs cargados exitosamente
        failed_ids: list de IDs que fallaron
    """

    if alff_dir is None:
        if config is not None:
            alff_dir = config.ALFF_DIR
        else:
            raise ValueError("Proporciona alff_dir o configura config.py")

    print(f"[INFO] Cargando ALFF desde: {alff_dir}")

    # Listar archivos
    nii_files = [f for f in os.listdir(alff_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]

    print(f"[INFO] Archivos ALFF encontrados: {len(nii_files)}")

    # Cargar los volumenes
    volumes = {}
    loaded_ids = []
    failed_ids = []
    skipped = []

    for nii_file in sorted(nii_files):
        subject_id = _extract_id_from_filename(nii_file)

        if subject_id is None:
            skipped.append(nii_file)
            continue

        if metadata_df is not None and subject_id not in metadata_df['ID'].values:
            continue

        try:
            volume = load_single_alff(os.path.join(alff_dir, nii_file))
            volumes[subject_id] = volume
            loaded_ids.append(subject_id)

        except Exception as e:
            failed_ids.append(subject_id)
            if len(failed_ids) <= 5:
                print(f"[WARN] Error cargando ALFF {subject_id}: {e}")

        # Progreso cada 500 archivos
        total_processed = len(loaded_ids) + len(failed_ids) + len(skipped)
        if total_processed % 500 == 0:
            print(f" ... {total_processed}/{len(nii_files)} procesados")

    if skipped:
        print(f"[WARN] {len(skipped)} archivos sin ID reconocible")
    if failed_ids:
        print(f"[WARN] {len(failed_ids)} archivos fallidos")

    print(f"[INFO] ALFF cargados: {len(loaded_ids)} sujetos\n")

    return volumes, loaded_ids, failed_ids

def load_all_data(metadata_path = None, roi_dir = None, alff_dir = None):
    """
    Carga todos los datos crudos del proyecto.

    Carga metadata, ROISignals y ALFF, y filtra para quedarse solo
    con los sujetos que tienen AMBOS tipos de datos disponibles.

    Returns:
        metadata: DataFrame filtrado (solo sujetos con datos completos)
        roi_signals: dict {subject_id: np.ndarray (timepoints, 116)}
        alff_volumes: dict {subject_id: np.ndarray (61, 73, 61)}
        common_ids: list de IDs que tienen metadata + ROI + ALFF
    """

    # 1. Metadata
    print('Cargando metadata...')
    metadata = load_metadata(metadata_path)

    # 2. ROIsignals
    print('Cargando ROISignals...')
    roi_signals, roi_ids, roi_failed = load_roi_signals(roi_dir, metadata_df=metadata)

    # 3. ALFF
    print('Cargando ALFF volumes...')
    alff_volumes, alff_ids, alff_failed = load_alff_volumes(alff_dir, metadata_df=metadata)

    # Sujetos que cuentan con las 3 cosas
    roi_set = set(roi_ids)
    alff_set = set(alff_ids)
    meta_set = set(metadata['ID'].values)

    common_ids = sorted(roi_set & alff_set & meta_set)

    # Filtrar metadata a solo sujetos completos
    metadata = metadata[metadata['ID'].isin(common_ids)].copy()
    metadata = metadata.set_index('ID').loc[common_ids].reset_index()

    print(f"  Sujetos en metadata:   {len(meta_set)}")
    print(f"  Sujetos con ROI:       {len(roi_set)}")
    print(f"  Sujetos con ALFF:      {len(alff_set)}")
    print(f"  Sujetos completos:     {len(common_ids)}")
    print(f"    MDD: {metadata['label'].sum()}")
    print(f"    HC:  {(metadata['label'] == 0).sum()}")

    return metadata, roi_signals, alff_volumes, common_ids

# Ejecucion testing
if __name__ == '__main__':
    """
    Ejecutar para verificar que todo carga bien:
        python src/data_loader.py
    """
    try:
        metadata, roi_signals, alff_volumes, common_ids = load_all_data()

        # Mostrar ejemplo de un sujeto
        ejemplo = common_ids[0]
        print(f"\nSujeto ejemplo: {ejemplo}")
        print(f"  ROI shape:  {roi_signals[ejemplo].shape}")
        print(f"  ALFF shape: {alff_volumes[ejemplo].shape}")
        info = parse_subject_id(ejemplo)
        print(f"  Sitio: {info['site']}, Label: {info['label']}")

    except FileNotFoundError as e:
        print(f"\n[ERROR] Archivo no encontrado: {e}")
        print("Verifica que las rutas en config.py sean correctas.")
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")

    

    
    

