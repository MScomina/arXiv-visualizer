import os
from pandas import read_json
from embedder import embed_batch
from transformers import AutoTokenizer, AutoModel

CHUNK_SIZE = 10_000
BATCH_SIZE = 32
DATASET_PATH = "./data/arxiv-metadata-oai-snapshot.json"
PROCESSED_FILE = "./data/processed/embeddings.json"

def main():

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

    with open(PROCESSED_FILE, "w", encoding="utf-8") as out:

        for chunk in read_json(DATASET_PATH, lines=True, chunksize=CHUNK_SIZE):

            abstracts = chunk["abstract"].tolist()

            embeddings = embed_batch(
                abstracts, 
                tokenizer, 
                model,
                batch_size=BATCH_SIZE
            )

            chunk["embedding"] = embeddings.tolist()
            chunk[["id","embedding"]].to_json(out, orient="records", lines=True)

if __name__ == "__main__":
    main()