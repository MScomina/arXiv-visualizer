import numpy as np
import torch
import torch.nn.functional as F
import hashlib
import pandas as pd
from autoencoder import AE

import logging
import os
import sys
from torch.utils.data import TensorDataset, DataLoader, random_split, IterableDataset

RANDOM_STATE = 31412718

EPOCHS = 250
BATCH_SIZE = 2048
TRAINING_RATE = 0.0005
DROPOUT = 0.1
GAMMA = 0.99
WEIGHT_DECAY = 1e-5
TRAIN_VAL_TEST_SPLIT = (0.8, 0.1, 0.1)

IN_DIMENSIONS = 768
HIDDEN_LAYERS = (512, 512, 384, 384, 256, 256, 192, 192, 128, 128)
LATENT_SPACE = 64
CHUNKSIZE = BATCH_SIZE*16

DATASET_PATH = "./data/processed/embeddings.json"
LOG_FILE = f"./models/ae - {HIDDEN_LAYERS} - {LATENT_SPACE}.log"
MODEL_FILE = f"./models/ae - {HIDDEN_LAYERS} - {LATENT_SPACE}.pt"
N_ROWS = 2200000    # Lower this number if you're having problems with RAM.

PRINT_EVERY = 25

class SequentialJsonlDataset(IterableDataset):

    def __init__(self, path: str, dtype: torch.dtype = torch.float32,
                 chunk_size: int = CHUNKSIZE, max_rows: int | None = N_ROWS):
        self.path = path
        self.dtype = dtype
        self.chunk_size = chunk_size
        self.max_rows = max_rows

    def __iter__(self):

        rows_yielded = 0
        for chunk in pd.read_json(self.path, lines=True,
                                  chunksize=self.chunk_size,
                                  nrows=self.max_rows):
            # ``chunk["embedding"]`` is a Series of lists
            for emb in chunk["embedding"]:
                yield torch.tensor(emb, dtype=self.dtype)
                rows_yielded += 1
                if self.max_rows is not None and rows_yielded >= self.max_rows:
                    return

def _compute_stats(path: str = DATASET_PATH,
                   dtype: torch.dtype = torch.float32,
                   chunk_size: int = CHUNKSIZE,
                   ratios: tuple[float, float, float] = TRAIN_VAL_TEST_SPLIT) -> tuple[torch.Tensor, torch.Tensor]:

    stats_file = f"{path}.stats.pt"
    if os.path.exists(stats_file):
        cached = torch.load(stats_file)
        return cached["mean"], cached["std"]

    total_sum, total_sumsq = torch.tensor(0., dtype=dtype), torch.tensor(0., dtype=dtype)
    total_n = 0

    thresholds = np.cumsum(np.array(ratios))
    def _bucket(idx: int) -> str:
        rnd_bytes = hashlib.sha256(str(idx).encode()).digest()
        rnd_val = int.from_bytes(rnd_bytes[:8], "big") / 2**64

        if rnd_val < thresholds[0]:
            return "train"
        else:
            return None

    idx = 0
    for chunk in pd.read_json(path, lines=True, chunksize=chunk_size):
        for emb in chunk["embedding"]:
            if _bucket(idx) != "train":
                idx += 1
                continue
            idx += 1
            vec = torch.tensor(emb, dtype=dtype)
            total_sum   += vec.sum()
            total_sumsq += (vec ** 2).sum()
            total_n     += vec.numel()

    mean = total_sum / total_n
    var  = total_sumsq / total_n - mean.pow(2)
    std  = torch.sqrt(var.clamp_min(1e-8))

    # Save computed statistics for future runs
    torch.save({"mean": mean, "std": std}, stats_file)

    return mean, std

    idx = 0
    for chunk in pd.read_json(path, lines=True, chunksize=chunk_size):
        for emb in chunk["embedding"]:
            if _bucket(idx) != "train":
                idx += 1
                continue
            idx += 1
            vec = torch.tensor(emb, dtype=dtype)
            total_sum   += vec.sum()
            total_sumsq += (vec ** 2).sum()
            total_n     += vec.numel()

    mean = total_sum / total_n
    var  = total_sumsq / total_n - mean.pow(2)
    std  = torch.sqrt(var.clamp_min(1e-8))
    return mean, std


def _normalize(batch : torch.Tensor, mean, std) -> torch.Tensor:
    return (batch - mean) / std


def create_loaders_from_file(
    path: str,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    batch_size: int = 256,
    dtype: torch.dtype = torch.float32,
    n_rows: int | None = None,
) -> dict[str, DataLoader]:

    embeddings = []
    for chunk in pd.read_json(path, lines=True,
                              chunksize=CHUNKSIZE,
                              nrows=n_rows if n_rows is not None else N_ROWS):
        embeddings.append(torch.tensor(chunk["embedding"].values.tolist(),
                                       dtype=dtype))
    data_tensor = torch.cat(embeddings, dim=0)

    train_idx, val_idx, test_idx = split_indices(len(data_tensor), ratios)

    train_ds = TensorDataset(data_tensor[train_idx])
    val_ds   = TensorDataset(data_tensor[val_idx])
    test_ds  = TensorDataset(data_tensor[test_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return {"train": train_loader, "val": val_loader, "test": test_loader}



def get_data_loaders(data: torch.Tensor,
                     ratios: tuple[float, float, float],
                     batch_size: int,
                     dtype: torch.dtype = torch.float32):

    train_idx, val_idx, test_idx = split_indices(len(data), ratios)

    train_ds = TensorDataset(data[train_idx])
    val_ds   = TensorDataset(data[val_idx])
    test_ds  = TensorDataset(data[test_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return {"train": train_loader, "val": val_loader, "test": test_loader}

def split_indices(num_samples: int, ratios: tuple[float, float, float], seed: int = RANDOM_STATE):

    indices = np.arange(num_samples)
    np.random.seed(seed)
    np.random.shuffle(indices)
    n_train = int(ratios[0] * num_samples)
    n_val   = int(ratios[1] * num_samples)
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train+n_val]
    test_idx  = indices[n_train+n_val:]
    return train_idx, val_idx, test_idx

def train_epoch_dl(autoencoder, loader, optimizer, loss_fn, device, logger):

    autoencoder.train()
    epoch_loss, n_samples = 0.0, 0
    batch_cnt = 0

    for batch in loader:
        batch = batch[0].to(device)
        x = batch

        optimizer.zero_grad()
        recon = autoencoder(x)
        loss  = loss_fn(recon, x)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * x.size(0)
        n_samples  += x.size(0)
        batch_cnt  += 1

        if batch_cnt % PRINT_EVERY == 0:
            avg_loss = epoch_loss / n_samples
            logger.info(f"[{batch_cnt:>6d} train] "
                        f"avg train loss (so far) = {avg_loss:.8f}")

    return epoch_loss, n_samples


def eval_epoch_dl(autoencoder, loader, loss_fn, device):

    autoencoder.eval()
    epoch_loss, n_samples = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            batch = batch[0].to(device)
            x = batch
            recon = autoencoder(x)
            loss  = loss_fn(recon, x)
            epoch_loss += loss.item() * x.size(0)
            n_samples  += x.size(0)

    return epoch_loss, n_samples

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    autoencoder = AE(
        dropout=DROPOUT,
        in_dimensions=IN_DIMENSIONS,
        hidden_layers=HIDDEN_LAYERS,
        latent_size=LATENT_SPACE
    ).to(device=device)

    try:
        autoencoder.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
    except FileNotFoundError:
        pass

    optimizer = torch.optim.AdamW(
        autoencoder.parameters(),
        lr=TRAINING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer,
        gamma=GAMMA
    )

    huber_loss = torch.nn.HuberLoss(delta=0.5)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(LOG_FILE, mode="a"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)

    mean, std = _compute_stats()

    best_loss = float("inf")

    loaders = create_loaders_from_file(DATASET_PATH, TRAIN_VAL_TEST_SPLIT, BATCH_SIZE, n_rows=N_ROWS)

    for epoch in range(EPOCHS):
            
        epoch_loss, n_train = train_epoch_dl(autoencoder, loaders["train"],
                                                  optimizer, huber_loss, device, logger)
            
        val_loss, n_val = eval_epoch_dl(autoencoder, loaders["val"],
                                            huber_loss, device)
            
        test_loss, n_test = eval_epoch_dl(autoencoder, loaders["test"],
                                              huber_loss, device)

            
        if val_loss < best_loss and epoch > 0:
            logger.info(f"Saved weights to {MODEL_FILE}")
            torch.save(autoencoder.state_dict(), MODEL_FILE)
            best_loss = val_loss

        epoch_loss = epoch_loss  / n_train if n_train  else float("nan")
        val_loss   = val_loss    / n_val    if n_val    else float("nan")
        test_loss  = test_loss   / n_test   if n_test   else float("nan")

        logger.info(f"Epoch {epoch+1:02d} – "
                    f"train {epoch_loss:.8f} | "
                    f"val   {val_loss:.8f} | "
                    f"test  {test_loss:.8f}")

        scheduler.step()

if __name__ == "__main__":
    main()