"""
combat.py — Armonizacion de sitios con neuroCombat

El dataset REST-meta-MDD viene de 25 hospitales con distintos escaneres Sin armonizacion, el modelo puede aprender a detectar de que hospital viene el paciente en lugar de detectar depresion.

ComBat es un algoritmo estadistico bayesiano que elimina ese "efecto sitio"
mientras preserva las diferencias biologicas reales (edad, sexo, diagnostico).

Este modulo toma los features y genera versiones armonizadas con sufijo _combat.npy:
  fc_features.npy   -> fc_features_combat.npy
  alff_features.npy -> alff_features_combat.npy

Funciones:
  - load_features_and_meta() : Carga los .npy y la metadata alineada
  - get_site_vector()        : Extrae el vector de sitios desde los IDs
  - get_covariate_matrix()   : Construye matriz de covariables biologicas
  - run_combat()             : Aplica neuroCombat por chunks a una matriz
  - harmonize_all()          : Pipeline completo para FC y ALFF

Proyecto: Prediccion de depresion con IA -- Samsung Innovation Campus
Dataset: REST-meta-MDD Phase 1
"""

import os
import numpy as np
import pandas as pd
from data_loader import load_metadata

try:
    import config
except ImportError:
    print("[WARN] No se encontro config.py. Usando rutas por defecto.")
    config = None

try:
    from neuroCombat import neuroCombat
except ImportError:
    print("[ERROR] neuroCombat no esta instalado.")
    neuroCombat = None


def load_features_and_meta(processed_dir=None, metadata_path=None):
    """
    Carga los features extraidos y la metadata, alineados por subject_id.

    Los archivos .npy de features estan ordenados segun subject_ids.npy.
    La metadata se carga con load_metadata() de data_loader.py y luego
    se reordena para que cada fila corresponda exactamente con la fila
    del mismo indice en los arrays de features.

    Args:
        processed_dir: ruta a data/processed/ (usa config si no se da)
        metadata_path: ruta al Excel de metadata (usa config si no se da)

    Returns:
        X_fc:        array (n_sujetos, 1679028) -- features FC originales (mmap)
        X_alff:      array (n_sujetos, 49)      -- features ALFF originales
        subject_ids: array (n_sujetos,)          -- IDs en el orden de los arrays
        metadata:    DataFrame alineado con los arrays (fila i = sujeto i)
    """
    if processed_dir is None:
        if config is not None:
            processed_dir = config.PROCESSED_DIR
        else:
            raise ValueError("Proporciona processed_dir o configura config.py")

    print(f"[INFO] Cargando features desde: {processed_dir}")

    # FC con mmap_mode='r' para no cargarlo completo en RAM (son ~14 GB)
    X_fc   = np.load(os.path.join(processed_dir, 'fc_features.npy'),  mmap_mode='r')
    X_alff = np.load(os.path.join(processed_dir, 'alff_features.npy'))

    subject_ids = np.load(
        os.path.join(processed_dir, 'subject_ids.npy'),
        allow_pickle=True
    )

    print(f"[INFO] Features cargados:")
    print(f"  FC:          {X_fc.shape}")
    print(f"  ALFF:        {X_alff.shape}")
    print(f"  Subject IDs: {len(subject_ids)} sujetos")

    # Reusar load_metadata() de data_loader.py — no duplicamos logica
    metadata = load_metadata(metadata_path)

    # Reordenar metadata para que coincida con el orden de subject_ids
    metadata = metadata.set_index('ID').loc[subject_ids].reset_index()

    print(f"[INFO] Metadata alineada: {len(metadata)} sujetos")
    print(f"  MDD:           {metadata['label'].sum()}")
    print(f"  HC:            {(metadata['label'] == 0).sum()}")
    print(f"  Sitios unicos: {metadata['site'].nunique()}")

    return X_fc, X_alff, subject_ids, metadata


def get_site_vector(metadata):
    """
    Extrae el vector de sitios desde la metadata.

    ComBat llama a esto el "batch": la variable que identifica de que
    hospital/escaner viene cada sujeto. Es el efecto que queremos eliminar.

    El sitio ya esta en la columna 'site' del DataFrame (anadida por
    load_metadata()). Lo convertimos a entero: S1->1, S2->2, etc.

    Args:
        metadata: DataFrame alineado (salida de load_features_and_meta)

    Returns:
        site_vector: array (n_sujetos,) de enteros -- ej. [1, 1, 2, 2, 3, ...]
        site_map:    dict {nombre_sitio: entero} para referencia futura
    """
    site_names = metadata['site'].values

    # Ordenar numericamente: S1, S2, ..., S25
    unique_sites = sorted(set(site_names), key=lambda s: int(s[1:]))
    site_map = {name: idx + 1 for idx, name in enumerate(unique_sites)}

    site_vector = np.array([site_map[s] for s in site_names])

    print(f"\n[INFO] Vector de sitios generado ({len(unique_sites)} sitios):")
    for name, code in site_map.items():
        n = (site_vector == code).sum()
        print(f"  {name} (codigo {code:2d}): {n} sujetos")

    return site_vector, site_map


def get_covariate_matrix(metadata):
    """
    Construye la matriz de covariables biologicas a preservar.

    ComBat elimina el efecto del sitio pero PRESERVA las diferencias
    explicadas por estas covariables. Sin pasarle el diagnostico,
    podria eliminar parcialmente la diferencia MDD vs HC.

    Covariables:
      - label: diagnostico (1=MDD, 0=HC) -- lo mas importante de preservar
      - Age:   edad del sujeto
      - Sex:   sexo (1=M, 2=F segun el dataset)

    NaN en Age o Sex se imputan con mediana/moda para no perder sujetos.

    Args:
        metadata: DataFrame alineado (salida de load_features_and_meta)

    Returns:
        covariates: DataFrame con columnas [label, Age, Sex], sin NaN
    """
    covariates = metadata[['label', 'Age', 'Sex']].copy()

    nan_age = covariates['Age'].isna().sum()
    nan_sex = covariates['Sex'].isna().sum()

    if nan_age > 0:
        mediana = covariates['Age'].median()
        print(f"[WARN] {nan_age} sujetos sin edad -- imputando con mediana ({mediana:.1f})")
        covariates['Age'] = covariates['Age'].fillna(mediana)

    if nan_sex > 0:
        moda = covariates['Sex'].mode()[0]
        print(f"[WARN] {nan_sex} sujetos sin sexo -- imputando con moda ({int(moda)})")
        covariates['Sex'] = covariates['Sex'].fillna(moda)

    print(f"\n[INFO] Covariables biologicas a preservar:")
    print(f"  label -- MDD: {(covariates['label'] == 1).sum()}, HC: {(covariates['label'] == 0).sum()}")
    print(f"  Age   -- media={covariates['Age'].mean():.1f}, std={covariates['Age'].std():.1f}")
    print(f"  Sex   -- 1(M): {(covariates['Sex'] == 1).sum()}, 2(F): {(covariates['Sex'] == 2).sum()}")

    return covariates


def run_combat(X, site_vector, covariates, out_path, feature_name='features',
               chunk_size=3000):
    """
    Aplica neuroCombat por chunks de features y escribe el resultado
    directo a disco con np.memmap. Nunca mantiene el array completo en RAM.

    Args:
        X:            array (n_sujetos, n_features) — puede ser mmap de lectura
        site_vector:  array (n_sujetos,) de enteros — sitio de cada sujeto
        covariates:   DataFrame con columnas [label, Age, Sex]
        out_path:     ruta completa del archivo .npy de salida
        feature_name: nombre descriptivo para los prints (ej. 'FC', 'ALFF')
        chunk_size:   numero de features por iteracion (default 3000)

    Returns:
        out_path: la misma ruta del archivo escrito (para confirmar)
    """
    if neuroCombat is None:
        raise ImportError(
            "neuroCombat no esta instalado. Ejecuta: pip install neuroCombat"
        )

    n_sujetos, n_features = X.shape
    n_chunks = (n_features + chunk_size - 1) // chunk_size

    print(f"\n[INFO] Aplicando ComBat a {feature_name} por chunks...")
    print(f"  Sujetos:          {n_sujetos}")
    print(f"  Features:         {n_features:,}")
    print(f"  Sitios:           {len(np.unique(site_vector))}")
    print(f"  Chunk size:       {chunk_size:,} features")
    print(f"  Total chunks:     {n_chunks}")
    ram_est = chunk_size * n_sujetos * 8 * 4 / 1e6
    print(f"  RAM por chunk:    ~{ram_est:.0f} MB (estimado)")

    # Covariables — iguales para todos los chunks
    covars_df = pd.DataFrame({
        'site':  site_vector,
        'label': covariates['label'].values,
        'age':   covariates['Age'].values,
        'sex':   covariates['Sex'].values,
    })

    # Archivo temporal en disco para acumular resultados sin usar RAM
    # Usamos .dat porque memmap no escribe cabeceras .npy
    tmp_path = out_path.replace('.npy', '_tmp.dat')
    X_out = np.memmap(tmp_path, dtype=np.float32, mode='w+',
                      shape=(n_sujetos, n_features))

    print(f"[INFO] Archivo temporal en disco: {tmp_path}")
    print(f"[INFO] Procesando chunks...")

    for chunk_idx in range(n_chunks):
        f_start = chunk_idx * chunk_size
        f_end   = min(f_start + chunk_size, n_features)

        # Leer solo las columnas del chunk desde el mmap de entrada
        chunk = np.array(X[:, f_start:f_end], dtype=np.float64)

        combat_result = neuroCombat(
            dat=chunk.T,
            covars=covars_df,
            batch_col='site',
            continuous_cols=['age'],
            categorical_cols=['label', 'sex']
        )

        # Escribir resultado directamente al archivo en disco
        X_out[:, f_start:f_end] = combat_result['data'].T.astype(np.float32)

        # Progreso cada 50 chunks
        if (chunk_idx + 1) % 50 == 0 or (chunk_idx + 1) == n_chunks:
            pct = (chunk_idx + 1) / n_chunks * 100
            print(f"  ... chunk {chunk_idx + 1}/{n_chunks} "
                  f"({pct:.1f}%) -- features {f_start:,} a {f_end:,}")

    # Forzar escritura a disco
    X_out.flush()

    n_sample = min(100, n_features)
    sample_before = float(np.array(X[:, :n_sample], dtype=np.float64).mean())
    sample_after  = float(X_out[:, :n_sample].mean())
    print(f"\n[INFO] Verificacion (primeras {n_sample} features):")
    print(f"  Media antes:   {sample_before:.6f}")
    print(f"  Media despues: {sample_after:.6f}")

    del X_out

    # Convertir .dat a .npy leyendo en modo solo lectura
    print(f"[INFO] Guardando como .npy en: {out_path}")
    final_array = np.memmap(tmp_path, dtype=np.float32, mode='r',
                            shape=(n_sujetos, n_features))
    np.save(out_path, final_array)

    del final_array

    # Limpiar archivo temporal
    os.remove(tmp_path)
    print(f"[INFO] Archivo temporal eliminado.")
    print(f"[INFO] ComBat completado para {feature_name}.")

    return out_path


def harmonize_all(processed_dir=None, metadata_path=None):
    """
    Pipeline completo: carga features, armoniza FC y ALFF, guarda resultados.

    Lee:
      data/processed/fc_features.npy         (2293, 1679028)
      data/processed/alff_features.npy       (2293, 49)
      data/processed/subject_ids.npy         (2293,)

    Escribe:
      data/processed/fc_features_combat.npy   (2293, 1679028)
      data/processed/alff_features_combat.npy (2293, 49)
      data/processed/combat_site_map.npy      -- mapa sitio->entero

    Los archivos originales NO se modifican.

    Args:
        processed_dir: ruta a data/processed/ (usa config si no se da)
        metadata_path: ruta al Excel de metadata (usa config si no se da)
    """
    if processed_dir is None:
        if config is not None:
            processed_dir = config.PROCESSED_DIR
        else:
            raise ValueError("Proporciona processed_dir o configura config.py")

    # Paso 1: Cargar features y metadata
    X_fc, X_alff, subject_ids, metadata = load_features_and_meta(
        processed_dir=processed_dir,
        metadata_path=metadata_path
    )

    # Paso 2: Vector de sitios
    site_vector, site_map = get_site_vector(metadata)

    # Paso 3: Covariables biologicas
    covariates = get_covariate_matrix(metadata)

    # Rutas de salida
    fc_out   = os.path.join(processed_dir, 'fc_features_combat.npy')
    alff_out = os.path.join(processed_dir, 'alff_features_combat.npy')
    map_out  = os.path.join(processed_dir, 'combat_site_map.npy')

    # Paso 4: Armonizar FC por chunks
    print("  ARMONIZANDO FC (1,679,028 features)")
    run_combat(
        X=X_fc,
        site_vector=site_vector,
        covariates=covariates,
        out_path=fc_out,
        feature_name='FC',
        chunk_size=3000
    )

    # Paso 5: Armonizar ALFF
    print("  ARMONIZANDO ALFF (49 features)")
    run_combat(
        X=X_alff,
        site_vector=site_vector,
        covariates=covariates,
        out_path=alff_out,
        feature_name='ALFF',
        chunk_size=3000
    )

    # Paso 6: Guardar mapa de sitios para referencia
    np.save(map_out, site_map)

    print(f"\n[INFO] Archivos generados en: {processed_dir}")
    print(f"  fc_features_combat.npy   OK")
    print(f"  alff_features_combat.npy OK")
    print(f"  combat_site_map.npy      OK ({len(site_map)} sitios)")
    print("\n[INFO] Armonizacion completada.")


if __name__ == '__main__':
    """
    Ejecutar para armonizar los features por sitio:
        python src/combat.py

    Prerequisitos:
        1. Haber corrido feature_extraction.py
        2. pip install neuroCombat

    Salidas:
        data/processed/fc_features_combat.npy
        data/processed/alff_features_combat.npy
        data/processed/combat_site_map.npy
    """
    try:
        harmonize_all()

    except FileNotFoundError as e:
        print(f"\n[ERROR] Archivo no encontrado: {e}")
        print("  Verifica que hayas corrido feature_extraction.py primero.")
    except ImportError as e:
        print(f"\n[ERROR] {e}")
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()