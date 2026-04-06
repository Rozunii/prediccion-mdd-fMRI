"""
clustering.py -- Clustering de pacientes por perfil de sintomas PHQ-9

Agrupa los 185 pacientes segun su perfil promedio de sintomas depresivos
a lo largo de los 14 dias de evaluacion ambulatoria.

Pipeline:
  1. Agregar por user_id 
  2. Elbow + Silhouette para elegir K optimo
  3. K-Means
  4. UMAP para reduccion a 2D y visualizacion
  5. HDBSCAN sobre embedding UMAP
  6. Validacion
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from src import config
    from src.phq9.data_loader import cargar_datos, preprocesar_features, preparar_clustering
except ImportError:
    print('[WARN] No se encontro config.py')
    config = None

FIGURES_DIR = os.path.join(str(config.FIGURES_DIR) if config else 'results/figures', 'phq9')
METRICS_DIR = os.path.join(str(config.METRICS_DIR) if config else 'results/metrics', 'phq9')

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# Features usadas para clustering (perfil de sintomas)
CLUSTER_FEATURES = [f'phq{i}' for i in range(1, 10)] + ['happiness_mean', 'happiness_std']


def elegir_k(X: np.ndarray, k_min: int = 2, k_max: int = 8) -> int:
    """Evalua K-Means para distintos valores de K y grafica elbow + silhouette.

    Args:
        X
        k_min
        k_max

    Returns:
        K optimo segun silhouette score
    """
    inercias    = []
    silhouettes = []
    ks          = range(k_min, k_max + 1)

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inercias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    # Grafica combinada elbow + silhouette
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(ks, inercias, 'o-', color='steelblue')
    ax1.set_xlabel('Numero de clusters (K)')
    ax1.set_ylabel('Inercia')
    ax1.set_title('Elbow Method')
    ax1.set_xticks(list(ks))

    ax2.plot(ks, silhouettes, 'o-', color='coral')
    ax2.set_xlabel('Numero de clusters (K)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score por K')
    ax2.set_xticks(list(ks))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'clustering_elbow_silhouette.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'[INFO] Figura guardada: {path}')

    # K optimo = mayor silhouette
    k_optimo = k_min + int(np.argmax(silhouettes))
    print(f'[INFO] K optimo por silhouette: {k_optimo} (score={max(silhouettes):.3f})')

    return k_optimo


def aplicar_kmeans(X: np.ndarray, k: int) -> np.ndarray:
    """Aplica K-Means con el K elegido.

    Args:
        X
        k

    Returns:
        Array de etiquetas de cluster por usuario
    """
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    print(f'[INFO] K-Means K={k} | Silhouette={sil:.3f}')
    return labels


def aplicar_umap(X: np.ndarray) -> np.ndarray:
    """Reduce a 2D con UMAP para visualizacion.

    Args:
        X

    Returns:
        Embedding 2D (n_usuarios, 2)
    """
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(X)
    print(f'[INFO] UMAP embedding: {embedding.shape}')
    return embedding


def aplicar_hdbscan(embedding: np.ndarray) -> np.ndarray:
    """Aplica HDBSCAN sobre el embedding UMAP.

    HDBSCAN detecta clusters de densidad variable sin necesidad de
    especificar K.

    Args:
        embedding

    Returns:
        Array de etiquetas
    """
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
    labels = clusterer.fit_predict(embedding)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_ruido    = (labels == -1).sum()
    print(f'[INFO] HDBSCAN: {n_clusters} clusters, {n_ruido} outliers')
    return labels


def graficar_umap(embedding: np.ndarray, labels_kmeans: np.ndarray,
                  labels_hdbscan: np.ndarray) -> None:
    """Scatter UMAP coloreado por K-Means y por HDBSCAN lado a lado.

    Args:
        embedding
        labels_kmeans
        labels_hdbscan
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colores = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']

    # K-Means
    for k in np.unique(labels_kmeans):
        mask = labels_kmeans == k
        ax1.scatter(embedding[mask, 0], embedding[mask, 1],
                    c=colores[k % len(colores)], label=f'Cluster {k+1}',
                    alpha=0.7, s=40)
    ax1.set_title('UMAP + K-Means')
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    ax1.legend()

    # HDBSCAN
    unique_labels = np.unique(labels_hdbscan)
    for lbl in unique_labels:
        mask  = labels_hdbscan == lbl
        color = '#AAAAAA' if lbl == -1 else colores[lbl % len(colores)]
        name  = 'Ruido' if lbl == -1 else f'Cluster {lbl+1}'
        ax2.scatter(embedding[mask, 0], embedding[mask, 1],
                    c=color, label=name, alpha=0.7, s=40)
    ax2.set_title('UMAP + HDBSCAN')
    ax2.set_xlabel('UMAP 1')
    ax2.set_ylabel('UMAP 2')
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'clustering_umap.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'[INFO] Figura guardada: {path}')


def graficar_heatmap(df_agg: pd.DataFrame, labels: np.ndarray) -> None:
    """Heatmap de media de cada sintoma PHQ por cluster.

    Muestra que sintomas caracterizan a cada grupo de pacientes.

    Args:
        df_agg
        labels
    """
    phq_items = [f'phq{i}' for i in range(1, 10)]
    nombres_items = [
        'Anhedonia', 'Animo deprimido', 'Sueno', 'Energia',
        'Apetito', 'Autoestima', 'Concentracion', 'Psicomotor', 'Suicidio'
    ]

    df_plot = df_agg[phq_items].copy()
    df_plot.columns = nombres_items
    df_plot['cluster'] = labels + 1  # clusters 1-indexed para legibilidad

    medias = df_plot.groupby('cluster')[nombres_items].mean()

    fig, ax = plt.subplots(figsize=(11, 4))
    sns.heatmap(medias, annot=True, fmt='.2f', cmap='YlOrRd',
                vmin=0, vmax=3, ax=ax, linewidths=0.5)
    ax.set_title('Media de sintomas PHQ-9 por cluster')
    ax.set_xlabel('Sintoma')
    ax.set_ylabel('Cluster')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'clustering_heatmap_sintomas.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'[INFO] Figura guardada: {path}')


def validar_clusters(df_agg: pd.DataFrame, labels: np.ndarray) -> None:
    """Compara clusters en variables clinicas y demograficas.

    Imprime media de phq_total, happiness y distribucion de sexo
    por cluster para evaluar si los grupos tienen sentido clinico.

    Args:
        df_agg
        labels
    """
    df_val = df_agg.copy()
    df_val['cluster'] = labels + 1

    print(f'{"Cluster":<10} {"N":>4} {"PHQ medio":>10} {"Happiness":>10} {"Hap.Std":>8} {"Edad":>6}')
    print('-' * 54)

    for k in sorted(df_val['cluster'].unique()):
        sub = df_val[df_val['cluster'] == k]
        print(f'{k:<10} {len(sub):>4} {sub["phq_total_mean"].mean():>10.2f} '
              f'{sub["happiness_mean"].mean():>10.2f} '
              f'{sub["happiness_std"].mean():>8.2f} '
              f'{sub["age"].mean():>6.1f}')

    # Distribucion de sexo por cluster
    print('\nDistribucion de sexo por cluster:')
    print(df_val.groupby('cluster')['sex'].value_counts().unstack(fill_value=0))

    # Guardar resumen en JSON
    resumen = {}
    for k in sorted(df_val['cluster'].unique()):
        sub = df_val[df_val['cluster'] == k]
        resumen[f'cluster_{k}'] = {
            'n':              int(len(sub)),
            'phq_total_mean': round(float(sub['phq_total_mean'].mean()), 3),
            'happiness_mean': round(float(sub['happiness_mean'].mean()), 3),
            'happiness_std':  round(float(sub['happiness_std'].mean()), 3),
            'age_mean':       round(float(sub['age'].mean()), 1),
        }

    path = os.path.join(METRICS_DIR, 'clustering_kmeans_resumen.json')
    with open(path, 'w') as f:
        json.dump(resumen, f, indent=2)
    print(f'\n[INFO] Resumen guardado: {path}')


def graficar_boxplots(df_agg: pd.DataFrame, labels: np.ndarray) -> None:
    """Boxplots de phq_total y happiness por cluster.

    Args:
        df_agg
        labels
    """
    df_plot = df_agg.copy()
    df_plot['Cluster'] = [f'Cluster {l+1}' for l in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    df_plot.boxplot(column='phq_total_mean', by='Cluster', ax=ax1,
                    boxprops=dict(color='steelblue'),
                    medianprops=dict(color='red'))
    ax1.set_title('Score PHQ-9 por cluster')
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('PHQ total medio')
    plt.sca(ax1)
    plt.title('Score PHQ-9 por cluster')

    df_plot.boxplot(column='happiness_mean', by='Cluster', ax=ax2,
                    boxprops=dict(color='coral'),
                    medianprops=dict(color='red'))
    ax2.set_title('Felicidad por cluster')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Happiness score medio')
    plt.sca(ax2)
    plt.title('Felicidad por cluster')

    plt.suptitle('')
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'clustering_boxplots.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'[INFO] Figura guardada: {path}')


if __name__ == '__main__':
    # 1. Cargar y preprocesar
    df_raw = cargar_datos()
    df     = preprocesar_features(df_raw)

    # 2. Agregar por usuario
    X_cluster, user_ids, df_agg = preparar_clustering(df)
    print(f'[INFO] Matriz clustering: {X_cluster.shape} ({len(user_ids)} pacientes)')

    # 3. Elegir K optimo
    print('\n[INFO] Evaluando K optimo...')
    k_optimo = elegir_k(X_cluster, k_min=2, k_max=8)

    # 4. K-Means
    print(f'\n[INFO] Aplicando K-Means con K={k_optimo}...')
    labels_kmeans = aplicar_kmeans(X_cluster, k_optimo)

    # 5. UMAP
    print('\n[INFO] Calculando embedding UMAP...')
    embedding = aplicar_umap(X_cluster)

    # 6. HDBSCAN sobre UMAP
    print('\n[INFO] Aplicando HDBSCAN...')
    labels_hdbscan = aplicar_hdbscan(embedding)

    # 7. Figuras
    print('\n[INFO] Generando figuras...')
    graficar_umap(embedding, labels_kmeans, labels_hdbscan)
    graficar_heatmap(df_agg, labels_kmeans)
    graficar_boxplots(df_agg, labels_kmeans)

    # 8. Validacion clinica
    validar_clusters(df_agg, labels_kmeans)
