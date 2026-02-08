import random
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import jsonlines
import numpy as np
import faiss

from helper.data_loader import stream_dataset

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

COMPRESSED_PATH = os.getenv("COMPRESSED_PATH", "./data/processed/compressed_embeddings.sqlite3")
N_ROWS = int(os.getenv("N_PROCESSED_ROWS", 2914060))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE_FAISS", 200000))
EMBEDDING_DIMENSION = int(os.getenv("LATENT_SPACE", 64))

HNSW_M = 16
HNSW_EFCONSTRUCTION = 200
HNSW_DATASET_PERCENTAGE = 0.2

def build_faiss_index(path : str = COMPRESSED_PATH, n_rows : int = N_ROWS, chunk_size : int = CHUNK_SIZE) -> tuple[faiss.Index, dict[int, str]]:
    
    index = faiss.IndexHNSWFlat(EMBEDDING_DIMENSION, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EFCONSTRUCTION

    index_to_id = {}

    n_chunks = (n_rows + chunk_size - 1)//chunk_size
    for idx_chunk, chunk in enumerate(tqdm(stream_dataset(path, ["id", "embedding"], n_rows, chunk_size), 
                                        total=n_chunks,
                                        desc="Creating FAISS index")):

        for idx, entry in enumerate(chunk["id"]):
            index_to_id[idx_chunk*chunk_size+idx] = chunk["id"][idx]
        chunk = np.array(chunk["embedding"]).astype(np.float32)

        faiss.normalize_L2(chunk)

        index.add(chunk)

    return index, index_to_id