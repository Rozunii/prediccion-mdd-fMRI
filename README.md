# Prediccion de depresión mediante machine learning y fMRI

**Proyecto final para Samsung Innovation Campus**

Este proyecto utiliza técnicas de Machine Learning y Deep Learning para clasificar a pacientes con Trastorno Depresivo Mayor frente a Controles Sanos. El análisis se basa en datos de resonancia magnética funcional en estado de reposo (rs-fMRI) extraídos de **REST-meta-MDD**.

## Características del proyecto
Se evalúan y comparan tres enfoques principales de extracción de características cerebrales:
1. **FC (Conectividad Funcional):** Correlaciones entre series temporales de 1,833 regiones cerebrales.
2. **ALFF (Amplitud de Fluctuaciones de Baja Frecuencia):** Actividad regional calculada con el atlas de Harvard-Oxford (49 regiones).
3. **Combinado:** Fusión de características FC y ALFF.

## Estuctura del proyecto
\`\`\`text
depresion_mdd/
├── data/                   
├── results/
│   ├── figures/            # Gráficas generadas (Matrices de Confusión, ROC)
│   ├── metrics/            # Archivos .json con métricas (Sensibilidad, Especificidad)
│   └── models/             # (Ignorado en Git) Modelos entrenados (.pkl)
├── src/                    # Código fuente del proyecto
│   ├── data_loader.py      # Carga de datos del dataset REST-meta-MDD
│   ├── feature_extraction.py # Extracción de FC y ALFF
│   ├── preprocessing.py    # Estandarización, split de datos y reducción PCA
│   └── utils.py            # Herramientas base (métricas, guardado, gráficas)
├── config.py               # Variables globales y rutas del sistema
├── .gitignore              # Archivos excluidos de control de versiones
└── README.md               # Documentación principal
\`\`\`

## Estado actual
- [x] Extracción de características FC y ALFF implementada.
- [x] Preprocesamiento con estandarización, split y PCA desarrollado.
- [x] Utilidades de evalucion y visualización.
- [ ] Modelos de clasificación (SVM, Random Forest, MLP).
- [ ] Deep Learning (CNN).

## Agradecimientos y Creditos
Los datos de neuroimagen (rs-fMRI) e información fenotípica utilizados en este proyecto fueron obtenidos de la base de datos pública **REST-meta-MDD** (http://rfmri.org/REST-meta-MDD). 

Estos datos fueron compartidos por el **Consorcio DIRECT** (Depression Imaging REsearch ConsorTium) y el **International Big-Data Center for Depression Research, Institute of Psychology, Chinese Academy of Sciences**. 

En estricto cumplimiento con los términos de uso del proyecto, este repositorio protege la privacidad y confidencialidad de los participantes. Los datos crudos y la información fenotípica no se distribuyen en este repositorio público para evitar filtraciones a terceros.

**Proveedores de Datos / Autores del Consorcio:**
* The DIRECT Consortium

**Citas oficiales del dataset:**
1. Yan, C. G., Chen, X., Li, L., Castellanos, F. X., Bai, T. J., Bo, Q. J., ... & Zang, Y. F. (2019). *Reduced default mode network functional connectivity in patients with recurrent major depressive disorder*. Proc Natl Acad Sci U S A, 116(18): 9078-83.
2. Chen, X., Lu, B., Li, H. X., et al. (2022). *The DIRECT consortium and the REST-meta-MDD project: towards neuroimaging biomarkers of major depressive disorder*. Psychoradiology, Volume 2, Issue 1, Pages 32–42.

---
*Desarrollado por MindSeek Team - Samsung Innovation Campus*