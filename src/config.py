from pathlib import Path

# Rutas

# Raiz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Datos
DATA_DIR        = PROJECT_ROOT / "data"
RAW_DIR         = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"

# Dataset principal
DATASET_DIR     = RAW_DIR / "REST-meta-MDD-Phase1-Sharing"
RESULTS_DIR     = DATASET_DIR / "Results"

# Carpetas especificas que usamos
ROI_DIR         = RESULTS_DIR / "ROISignals_FunImgARCWF"
ALFF_DIR        = RESULTS_DIR / "ALFF_FunImgARglobalCW"

# Datos clinicos
METADATA_FILE   = RAW_DIR / "REST-meta-MDD-PhenotypicData_WithHAMDSubItem_V4.xlsx"

# Salidas del proyecto
OUTPUT_DIR      = PROJECT_ROOT / "results"
FIGURES_DIR     = OUTPUT_DIR / "figures"
METRICS_DIR     = OUTPUT_DIR / "metrics"
MODELS_DIR      = OUTPUT_DIR / "models"

# Parametros del dataset
N_ROIS = 1833  # Regiones del atlas AAL
N_FC_FEATURES = 1679028  # Triangulo superior de la matriz

# Etiquetas de diagnostico en el archivo de datos
LABEL_MDD       = 1
LABEL_HC        = 0

# Parametros de preprocesamiento
TEST_SIZE       = 0.15      # 15% reservado para prueba final (no tocar hasta el final)
VAL_SIZE        = 0.15      # 15% para validacion durante desarrollo
TRAIN_SIZE      = 0.70      # 70% para entrenamiento
RANDOM_STATE    = 42        # Semilla para reproducibilidad
APPLY_PCA       = True      # Reduccion de dimensionalidad antes del MLp
PCA_COMPONENTS  = 100       # Numero de componentes a conservar con PCA

# Verificacion
if __name__ == "__main__":
    rutas = {
        "Proyecto":         PROJECT_ROOT,
        "Dataset":          DATASET_DIR,
        "ROISignals":       ROI_DIR,
        "ALFF":             ALFF_DIR,
        "Metadatos":        METADATA_FILE,
        "Procesados":       PROCESSED_DIR,
    }

    print("Verificando rutas del proyecto:\n")
    for nombre, ruta in rutas.items():
        estado = "OK" if ruta.exists() else "NO ENCONTRADA"
        print(f"  [{estado}] {nombre}: {ruta}")