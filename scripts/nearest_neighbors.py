import random
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import jsonlines
import numpy as np
import faiss

from helper.data_loader import load_dataset

DATASET_PATH = "./data/arxiv-metadata-oai-snapshot.json"
COMPRESSED_PATH = "./data/processed/compressed_embeddings.json"
N_ROWS = 3_000_000

HNSW_M = 16
HNSW_EFCONSTRUCTION = 200
HNSW_DATASET_PERCENTAGE = 0.2
N_NEIGHBORS = 5

SEED = 314271

def fetch_abstracts_for_ids(file_path, ids_to_fetch: set):
    results = {}
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            if obj["id"] in ids_to_fetch:
                results[obj["id"]] = obj["abstract"]
                if len(results) == len(ids_to_fetch):
                    break
    return results

def main():

    np.random.seed(SEED)
    compressed_dataset = load_dataset(COMPRESSED_PATH, ["id", "embedding"], N_ROWS)
    id_list, embedding_matrix = np.array(compressed_dataset["id"]), np.array(compressed_dataset["embedding"]).astype(np.float32)
    
    faiss.normalize_L2(embedding_matrix)

    index = faiss.IndexHNSWFlat(embedding_matrix.shape[1], HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EFCONSTRUCTION
    index.add(embedding_matrix)

    similarities, indices = index.search(embedding_matrix, N_NEIGHBORS + 1)

    needed_ids = set()

    for i in range(400, 405):
        src_id = id_list[i]
        for nb_idx in indices[i]:
            needed_ids.add(id_list[nb_idx])

    id_to_abstract = fetch_abstracts_for_ids(DATASET_PATH, needed_ids)

    for i in range(400, 405):
        src_id = id_list[i]
        neighbor_ids = [id_list[idx] for idx in indices[i]
                         if id_list[idx] != src_id]
        print(f"\nSource ID: {src_id}")
        src_abstract = id_to_abstract.get(src_id, "No abstract available")
        print(f"\nSource Abstract: {src_abstract[:200]}")
        print("Neighbors:")
        for k, nb_id in enumerate(neighbor_ids):
            similarity = similarities[i][k + 1]
            abstract = id_to_abstract.get(nb_id, "No abstract available")
            print(f"Cosine Similarity: {similarity:.4f}")
            print(f"  ID: {nb_id}\n  Abstract: {abstract[:200]}...\n")

if __name__ == "__main__":
    main()