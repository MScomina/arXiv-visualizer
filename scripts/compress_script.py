import os
import json
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from autoencoder.autoencoder import AE

IN_DIMENSIONS = 768
HIDDEN_LAYERS = (512, 512, 384, 384, 256, 256, 192, 192, 128, 128)
LATENT_SPACE = 64
BATCH_SIZE = 2048
CHUNK_SIZE = BATCH_SIZE*16

DATASET_PATH = "./data/processed/embeddings.json"
AUTOENCODER_PATH = f"./models/ae - {HIDDEN_LAYERS} - {LATENT_SPACE}.pt"
COMPRESSED_PATH = "./data/processed/compressed_embeddings.json"

def _normalize(batch : torch.Tensor, mean, std) -> torch.Tensor | None:
    return (batch - mean) / std

def _load_stats(path: str = DATASET_PATH):
    stats_file = f"{path}.stats.pt"
    if os.path.exists(stats_file):
        cached = torch.load(stats_file)
        return cached["mean"], cached["std"]
    else:
        print("Mean and std file not found, please run training_script.py first.")
        return

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
    except FileNotFoundError:
        print("Model not found, run training_script.py first. Exiting.")
        return None

    model.eval()
    return model

def stream_embeddings(file_path):
    with open(file_path, "r") as f:
        for line in f:
            yield json.loads(line)

def chunker(generator, size):
    chunk = []
    for item in generator:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

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
    stream_gen = stream_embeddings(DATASET_PATH)

    with open(COMPRESSED_PATH, "w") as out_file:
        for chunk in chunker(stream_gen, CHUNK_SIZE):
            batch_embeddings = torch.tensor(
                [item["embedding"] for item in chunk],
                dtype=torch.float32,
                device=device,
            )
            mean, std = _load_stats()

            batch_embeddings = _normalize(batch_embeddings, mean, std)

            with torch.no_grad():
                latent = model.encoder(batch_embeddings).cpu().numpy()

            for record, vector in zip(chunk, latent):
                out_record = {"id": record["id"], "embedding": vector.tolist()}
                out_file.write(json.dumps(out_record) + "\n")

if __name__ == "__main__":
    main()