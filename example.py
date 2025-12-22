import pandas as pd
from helper.embedder import embed_batch
from transformers import AutoTokenizer, AutoModel

def main():

    total_size = 0
    for chunk in pd.read_json("./data/arxiv-metadata-oai-snapshot.json", lines=True, chunksize=100_000):
        total_size += chunk["abstract"].size
        print(total_size)

    #tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", device_map="cuda")
    #model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased", device_map="cuda")
    #model.eval()

    #all_vecs = embed_batch(
    #    df["abstract"].tolist(), 
    #    tokenizer, 
    #    model
    #)

    #print(all_vecs.shape)

if __name__ == "__main__":
    main()