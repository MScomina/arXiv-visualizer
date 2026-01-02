import umap
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATASET_PATH = "./data/processed/embeddings.json"
COMPRESSED_PATH = "./data/processed/compressed_embeddings.json"

def sample_dataset(file_path, sample_size=10000):
    df = pd.read_json(file_path, lines=True, nrows=sample_size)
    return df["embedding"].tolist()

def main():
    samples = np.array(sample_dataset(DATASET_PATH, sample_size=50000))
    samples = StandardScaler().fit_transform(samples)

    compressed_samples = np.array(sample_dataset(COMPRESSED_PATH, sample_size=50000))
    compressed_samples = StandardScaler().fit_transform(compressed_samples)

    umap_normal = umap.UMAP(
        n_neighbors=100,
        min_dist=0.01,
        metric='euclidean',
        n_components=2
    )
    umap_compressed = umap.UMAP(
        n_neighbors=100,
        min_dist=0.01,
        metric='euclidean',
        n_components=2
    )

    reduced = umap_normal.fit_transform(samples)
    compressed_reduced = umap_compressed.fit_transform(compressed_samples)

    labels_normal = np.zeros(reduced.shape[0], dtype=int)          # all zeros
    labels_compressed = np.ones(compressed_reduced.shape[0], dtype=int)  # all ones

    labels = np.concatenate([labels_normal, labels_compressed])

    all_points = np.vstack([reduced, compressed_reduced])

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        all_points[:, 0], all_points[:, 1],
        c=labels,
        cmap='tab10',
        s=5,
        alpha=0.6
    )
    plt.title('UMAP projection of SciBERT vs 64‑D compressed embeddings')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    cbar = plt.colorbar(scatter, ticks=[0, 1])
    cbar.set_ticklabels(['Normal SciBERT', 'Compressed 64‑D'])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()