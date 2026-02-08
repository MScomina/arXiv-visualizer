import os
import sys
import json
import torch
import sqlite3
import numpy as np
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from autoencoder.autoencoder import AE
from helper.data_loader import stream_dataset
from dotenv import load_dotenv
import ast

load_dotenv()

IN_DIMENSIONS = int(os.getenv("IN_DIMENSIONS", 768))
HIDDEN_LAYERS = ast.literal_eval(
    os.getenv("HIDDEN_LAYERS", "(512, 512, 384, 384, 256, 256, 192, 192, 128, 128)")
)
LATENT_SPACE = int(os.getenv("LATENT_SPACE", 64))
BATCH_SIZE = int(os.getenv("BATCH_SIZE_COMP", 2048))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE_COMP", str(BATCH_SIZE*16)))
N_ROWS = int(os.getenv("N_PROCESSED_ROWS", 2914060))

DATASET_PATH = os.getenv("EMBEDDINGS_PATH", "./data/processed/embeddings.parquet")
COMPRESSED_PATH = os.getenv("COMPRESSED_PATH", "./data/processed/compressed_embeddings.sqlite3")
AUTOENCODER_PATH = f"{os.getenv("AUTOENCODER_PATH", f"./models/ae - {HIDDEN_LAYERS} - {LATENT_SPACE}")}.pt"

def _normalize(batch : torch.Tensor, mean, std) -> torch.Tensor | None:
    return (batch - mean) / std

def _load_stats(path: str = DATASET_PATH):
    stats_file = f"{path}.stats.pt"
    if os.path.exists(stats_file):
        cached = torch.load(stats_file)
        return cached["mean"], cached["std"]
    else:
        raise FileNotFoundError("Mean and std file not found, please run training_ae_script.py first.")

def load_autoencoder():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AE(
        dropout=0.0,
        in_dimensions=IN_DIMENSIONS,
        hidden_layers=HIDDEN_LAYERS,
        latent_size=LATENT_SPACE,
    ).to(device=device)

    try:
        model.load_state_dict(torch.load(AUTOENCODER_PATH, weights_only=True))
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Model not found at '{AUTOENCODER_PATH}'. "
            "Run training_script.py first."
        ) from e

    model.eval()
    return model

def main():
    if os.path.exists(COMPRESSED_PATH):
        response = input("Compressed embeddings already exist, regenerate them? (y to proceed): ").strip().lower()
        if response != "y":
            print("Exiting...")
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_autoencoder()
    if model is None:
        return

    model = model.to(device)

    conn = sqlite3.connect(COMPRESSED_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS data (
            id          TEXT PRIMARY KEY,
            embedding   TEXT,          -- JSON string representation of embedding list
            abstract    TEXT,
            title       TEXT
        )
    """)
    conn.commit()

    total_chunks = (N_ROWS + CHUNK_SIZE - 1) // CHUNK_SIZE

    for chunk in tqdm(stream_dataset(DATASET_PATH, ["id", "title", "embedding", "abstract"], N_ROWS, CHUNK_SIZE), 
                    total=total_chunks,
                    desc="Compressing embeddings"):
        batch_embeddings = torch.tensor(
            chunk["embedding"],
            dtype=torch.float32,
            device=device,
        )
        mean, std = _load_stats()
        mean = mean.to(device)
        std = std.to(device)
        batch_embeddings = _normalize(batch_embeddings, mean, std)

        with torch.no_grad():
            latent = model.encoder(batch_embeddings).cpu().numpy().astype(np.float16)

        # Prepare rows for batch insertion
        rows = [
            (
                id_,
                json.dumps(vector.tolist()),  # Convert embedding to JSON string
                abstract,
                title
            )
            for id_, vector, abstract, title in zip(chunk["id"], latent, chunk["abstract"], chunk["title"])
        ]

        cursor.executemany(
            "INSERT OR REPLACE INTO data (id, embedding, abstract, title) VALUES (?, ?, ?, ?)",
            rows
        )
        conn.commit()

    conn.close()

    print(f"Compressed embeddings saved to SQLite database at {COMPRESSED_PATH}.")

if __name__ == "__main__":
    main()