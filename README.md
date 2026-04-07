# Detección de Depresión Mayor con fMRI y Machine Learning

**Proyecto final para Samsung Innovation Campus**

Este proyecto aplica Machine Learning y Deep Learning para clasificar pacientes con Trastorno Depresivo Mayor (MDD) frente a Controles Sanos (HC) usando datos de resonancia magnética funcional en estado de reposo (rs-fMRI) del dataset **REST-meta-MDD**. Adicionalmente, incluye análisis de regresión y clustering sobre síntomas PHQ-9 recolectados mediante Evaluación Momentánea Ecológica (EMA).

---

## Datasets

### REST-meta-MDD (fMRI)
- 2,293 sujetos: 1,186 MDD y 1,107 HC
- 25 sitios hospitalarios (China)
- Features extraídas: FC (Conectividad Funcional) y ALFF

### PHQ-9 Ambulatory Assessment (EMA)
- 185 pacientes, ~16,150 observaciones durante 14 días
- Variables: ítems PHQ-9, felicidad diaria, demográficos
- Tareas: regresión de severidad y clustering de subtipos

---

## Estructura del proyecto

```text
depresion_mdd/
├── data/
│   ├── raw/                        # Datos crudos (.mat, NIfTI, PHQ-9 CSV)
│   └── processed/                  # Arrays preprocesados (X_train/val/test_*.npy)
│
├── src/
│   ├── config.py                   # Rutas e hiperparámetros globales
│   ├── data_loader.py              # Carga y validación de datos fMRI
│   ├── feature_extraction.py       # Extracción de FC (1,679,028 features) y ALFF (49)
│   ├── combat.py                   # Harmonización ComBat para corrección de sitio
│   ├── preprocessing.py            # PCA, ANOVA, mRMR, splits 70/15/15
│   ├── utils.py                    # Métricas, matrices de confusión, curvas ROC
│   │
│   ├── ml_clasico/
│   │   ├── train_svm.py            # SVM con RBF kernel (14 experimentos)
│   │   ├── train_random_forest.py  # Random Forest (14 experimentos)
│   │   └── train_xgboost.py        # XGBoost (14 experimentos)
│   │
│   ├── dl/
│   │   └── train_mlp.py            # MLP con K-Fold CV (TensorFlow/Keras)
│   │
│   └── phq9/
│       ├── data_loader.py          # Carga y preprocesamiento del dataset PHQ-9
│       ├── regression.py           # Ridge, SVR, XGBoost para predecir PHQ total
│       └── clustering.py          # K-Means, UMAP, HDBSCAN por perfil de síntomas
│
├── notebooks/
│   ├── 00_introduccion.ipynb       # Contexto clínico, datasets, pipeline, limitaciones
│   ├── 01_classic_ml.ipynb         # SVM, RF, XGB — 14 experimentos, efecto ComBat
│   ├── 02_regression_phq9.ipynb    # Regresión PHQ-9: Ridge, SVR, XGBoost
│   ├── 03_clustering_phq9.ipynb    # Clustering: K-Means, UMAP, HDBSCAN
│   └── 04_deep_learning.ipynb      # MLP con K-Fold CV vs modelos clásicos
│
├── results/
│   ├── figures/
│   │   ├── svm/                    # Matrices de confusión y curvas ROC del SVM
│   │   ├── rf/                     # Idem para Random Forest
│   │   ├── xgb/                    # Idem para XGBoost
│   │   ├── mlp/                    # Curvas de entrenamiento, CM, ROC del MLP
│   │   └── phq9/                   # Scatter plots, feature importance, clustering
│   ├── metrics/
│   │   ├── svm/                    # JSON de métricas por experimento
│   │   ├── rf/
│   │   ├── xgb/
│   │   ├── mlp/
│   │   └── phq9/                   # Métricas de regresión y resumen de clustering
│   └── models/                     # Modelos entrenados (.pkl, .keras)
│
├── .gitignore
└── README.md
```

---

## Pipeline fMRI

```bash
# 1. Cargar y validar datos crudos
python src/data_loader.py

# 2. Extraer features (requiere archivos .mat y NIfTI en data/raw/)
python src/feature_extraction.py

# 3. Harmonización ComBat (opcional)
python src/combat.py

# 4. Preprocesar y generar splits
python src/preprocessing.py              # 7 experimentos
python src/preprocessing.py combat       # 7 experimentos adicionales con ComBat

# 5. Entrenar clasificadores
python src/ml_clasico/train_svm.py
python src/ml_clasico/train_random_forest.py
python src/ml_clasico/train_xgboost.py
python src/dl/train_mlp.py
```

## Pipeline PHQ-9

```bash
python src/phq9/data_loader.py    # Limpieza y splits por user_id
python src/phq9/regression.py     # Ridge, SVR, XGBoost
python src/phq9/clustering.py     # K-Means + UMAP + HDBSCAN
```

---

## Agradecimientos y Créditos

Los datos de neuroimagen (rs-fMRI) utilizados fueron obtenidos de **REST-meta-MDD** (http://rfmri.org/REST-meta-MDD), compartidos por el **Consorcio DIRECT** y el **International Big-Data Center for Depression Research, Institute of Psychology, Chinese Academy of Sciences**.

En cumplimiento con los términos de uso, los datos crudos y la información fenotípica no se distribuyen en este repositorio.

**Citas del dataset:**
1. Yan, C. G., et al. (2019). *Reduced default mode network functional connectivity in patients with recurrent major depressive disorder*. PNAS, 116(18): 9078–83.
2. Chen, X., et al. (2022). *The DIRECT consortium and the REST-meta-MDD project: towards neuroimaging biomarkers of major depressive disorder*. Psychoradiology, 2(1): 32–42.

---

*Desarrollado por MindSeek Team — Samsung Innovation Campus*
