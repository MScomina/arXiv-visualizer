import os
import sys
import jsonlines
import pyarrow as pa
import pyarrow.parquet as pq

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from helper.embedder import embed_batch
from helper.data_loader import stream_dataset
from transformers import AutoTokenizer, AutoModel

from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

N_ROWS = int(os.getenv("N_PROCESSED_ROWS", 2914060))
BATCH_SIZE = int(os.getenv("BATCH_SIZE_EMB", 64))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE_EMB", BATCH_SIZE * 128))
DATASET_PATH = os.getenv("DATASET_PATH", "./data/arxiv-metadata-oai-snapshot.json")
PROCESSED_FILE = os.getenv("EMBEDDINGS_PATH", "./data/processed/embeddings.parquet")

def main():

    total_chunks = (N_ROWS + CHUNK_SIZE - 1) // CHUNK_SIZE

    if os.path.exists(PROCESSED_FILE):
        response = input("Semantic embeddings already exist, regenerate them? (y to proceed): ").strip().lower()
        if response != "y":
            print("Exiting...")
            return
    
    tokenizer = AutoTokenizer.from_pretrained(
        "allenai/scibert_scivocab_uncased", 
        device_map="cuda", 
        use_fast=True
    )

    model = AutoModel.from_pretrained(
        "allenai/scibert_scivocab_uncased", 
        device_map="cuda"
    )

    model.eval()

    os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)

    writer : pq.ParquetWriter | None = None

    for chunk in tqdm(
        stream_dataset(DATASET_PATH, ["id", "title", "abstract"], N_ROWS, CHUNK_SIZE),
        total=total_chunks,
        desc="Processing embeddings"
    ):

        embeddings = embed_batch(
            chunk["abstract"], 
            tokenizer, 
            model,
            batch_size=BATCH_SIZE
        )

        chunk["embedding"] = [e.tolist() for e in embeddings]

        table = pa.Table.from_pydict(chunk)

        if writer is None:
            writer = pq.ParquetWriter(
                PROCESSED_FILE,
                table.schema,
                compression="snappy",
                use_dictionary=True,
                write_statistics=True
            )

        writer.write_table(table)

    if writer is not None:
        writer.close()

if __name__ == "__main__":
    main()