import torch
import re
from torch.utils.data import DataLoader, Dataset

class Abstracts(Dataset):
    def __init__(self, abstracts):
        self.abstracts = abstracts
    def __len__(self):
        return len(self.abstracts)
    def __getitem__(self, idx):
        return self.abstracts[idx]

def _clean_abstract(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[citation\]', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\$[^\$]*\$', '', text)
    return text.strip()

def embed_batch(abstracts, tokenizer, model, batch_size=32, clean_abstracts=True):

    if clean_abstracts:
        abstracts = [_clean_abstract(abstract) for abstract in abstracts]

    loader = DataLoader(Abstracts(abstracts), batch_size=batch_size, shuffle=False)
    all_vecs = []

    for batch in loader:
        inputs = tokenizer(
            batch, 
            padding=True, 
            truncation=True,
            max_length=512, 
            return_tensors='pt'
        )
        inputs = {k:v.cuda() for k,v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        all_vecs.append(out.last_hidden_state[:, 0, :].cpu())

    return torch.cat(all_vecs, dim=0)