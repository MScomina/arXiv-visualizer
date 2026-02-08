import torch
from torch.utils.data import DataLoader, Dataset

import re

from tqdm import tqdm

class Abstracts(Dataset):
    def __init__(self, abstracts):
        self.abstracts = abstracts
    def __len__(self):
        return len(self.abstracts)
    def __getitem__(self, idx):
        return self.abstracts[idx]

def _clean_abstract(text):
    text = re.sub(r'\s+', ' ', text)

    text = re.sub(r'\\[a-zA-Z]+{[^}]*}', '', text)
    text = re.sub(r'\$[^$]+\$', '', text)
    text = re.sub(r'\\\[.*?\\\]', '', text)

    text = re.sub(r'(http\S+|www\S+|\b[\w.-]+@[\w.-]+\.\w{2,4}\b)', '', text)

    text = re.sub(r'\[\d+(-\d+)?(,\s*\d+(-\d+)?)*\]', '', text)
    text = re.sub(r'\([^\)]*et al.*\)', '', text, flags=re.I)

    text = re.sub(r'[–—]', '-', text)
    text = re.sub(r'[…]', '...', text)

    text = re.sub(r'[.]{2,}', '.', text)

    return text.strip()

def embed_batch(abstracts, tokenizer, model, batch_size=32, clean_abstracts=True, device="cuda"):

    if clean_abstracts:
        abstracts = [_clean_abstract(abstract) for abstract in abstracts]

    loader = DataLoader(
        Abstracts(abstracts),
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True,
        num_workers=4
        )

    all_vecs = []

    for batch in tqdm(loader, desc="Embedding batch", total=len(loader), leave=False):
        inputs = tokenizer(
            batch, 
            padding=True, 
            truncation=True,
            max_length=512, 
            return_tensors='pt'
        )
        inputs = {k:v.to(device) for k,v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        all_vecs.append(out.last_hidden_state[:, 0, :].cpu())

    return torch.cat(all_vecs, dim=0)