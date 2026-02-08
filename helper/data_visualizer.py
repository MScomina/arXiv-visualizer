import umap
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes

from data_loader import stream_dataset

import os
from dotenv import load_dotenv

load_dotenv()

DATASET_PATH = os.getenv("EMBEDDINGS_PATH", "./data/processed/embeddings.parquet")
COMPRESSED_PATH = os.getenv("COMPRESSED_PATH", "./data/processed/compressed_embeddings.sqlite3")

# UMAP parameters
N_SAMPLES = 50000
N_NEIGHBORS = 50

def align_points(X, Y):
    
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)

    R, _ = orthogonal_procrustes(Y_centered, X_centered)

    scale = np.linalg.norm(X_centered) / np.linalg.norm(Y_centered @ R)

    Y_aligned = (Y @ R) * scale + X.mean(axis=0)

    return Y_aligned, {"R": R, "scale": scale, "translation": X.mean(axis=0)}

def main():
    samples = np.array(next(stream_dataset(DATASET_PATH, ["embedding"], N_SAMPLES, N_SAMPLES))["embedding"])
    samples = StandardScaler().fit_transform(samples)

    compressed_samples = np.array(next(stream_dataset(COMPRESSED_PATH, ["embedding"], N_SAMPLES, N_SAMPLES))["embedding"])
    compressed_samples = StandardScaler().fit_transform(compressed_samples)

    umap_normal = umap.UMAP(
        n_neighbors=N_NEIGHBORS,
        min_dist=0.0,
        metric='cosine',
        n_components=2
    )
    umap_compressed = umap.UMAP(
        n_neighbors=N_NEIGHBORS,
        min_dist=0.0,
        metric='cosine',
        n_components=2
    )

    reduced = umap_normal.fit_transform(samples)
    compressed_reduced = umap_compressed.fit_transform(compressed_samples)

    compressed_aligned, params = align_points(reduced, compressed_reduced)

    plt.figure(figsize=(12, 5))

    # Left subplot – normal
    ax1 = plt.subplot(1, 2, 1)
    ax1.scatter(reduced[:, 0], reduced[:, 1], c='blue', s=5, alpha=0.6)
    ax1.set_title('Normal SciBERT')
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')

    # Right subplot – compressed (aligned)
    ax2 = plt.subplot(1, 2, 2)
    ax2.scatter(compressed_aligned[:, 0], compressed_aligned[:, 1], c='red', s=5, alpha=0.6)
    ax2.set_title('Compressed 64‑D (aligned)')
    ax2.set_xlabel('UMAP 1')
    ax2.set_ylabel('UMAP 2')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()