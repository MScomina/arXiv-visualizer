import numpy as np
import torch
import torch.nn.functional as F
import hashlib
import pandas as pd

import logging
import os
import sys
from torch.utils.data import DataLoader, random_split
from torch import Tensor

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from helper.data_loader import stream_dataset, EmbeddingStreamDataset, SubsetIterable
from autoencoder.autoencoder import AE

RANDOM_STATE = 31412718

EPOCHS = 25
TRAINING_RATE = 0.0002
DROPOUT = 0.1
GAMMA = 0.95
WEIGHT_DECAY = 1e-5
TRAIN_VAL_TEST_SPLIT = (0.8, 0.1, 0.1)

IN_DIMENSIONS = 768
HIDDEN_LAYERS = (512, 512, 384, 384, 256, 256, 192, 192, 128, 128)
LATENT_SPACE = 64

DATASET_PATH = "./data/processed/embeddings.parquet"
LOG_FILE = f"./models/ae - {HIDDEN_LAYERS} - {LATENT_SPACE}.log"
MODEL_FILE = f"./models/ae - {HIDDEN_LAYERS} - {LATENT_SPACE}.pt"

# Lower either of these numbers if you're having problems with RAM.
N_ROWS = 2_914_060
BATCH_SIZE = 16384
MINIBATCH_AMOUNT = 8
PREFETCH_FACTOR = 1

VAL_EVERY = 10

# Adjust this based on your computer specs.
NUM_WORKERS = 8

PRINT_EVERY = 25

def _chunk_collate(batch: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(batch, dim=0)

def _compute_stats(
    loader: DataLoader,
    stats_file : str,
    *,
    dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    if stats_file and os.path.exists(stats_file):
        cached = torch.load(stats_file)
        return cached["mean"], cached["std"]

    n_samples = 0
    mean = torch.zeros(0, dtype=torch.float32)
    M2 = torch.zeros(0, dtype=torch.float32)

    with torch.no_grad():
        for batch in loader:
            data = batch[0] if isinstance(batch, (tuple, list)) else batch
            B, D = data.shape
            if mean.numel() == 0:
                mean = torch.zeros(D, dtype=torch.float32)
                M2   = torch.zeros(D, dtype=torch.float32)

            vec = data.to(torch.float32)

            n_prev = n_samples
            n_samples += B
            delta = vec - mean.unsqueeze(0)
            mean += delta.sum(dim=0) / n_samples
            delta2 = vec - mean.unsqueeze(0)
            M2   += (delta * delta2).sum(dim=0)

    var = M2 / (n_samples - 1)
    std = torch.sqrt(var.clamp_min(1e-8))

    mean = mean.float()
    std  = std.float()

    if stats_file:
        torch.save({"mean": mean, "std": std}, stats_file)

    return mean, std

def _normalize(batch : torch.Tensor, mean, std) -> torch.Tensor:
    return (batch - mean) / std

def create_loaders(
    path: str,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    batch_size: int = BATCH_SIZE,
    dtype: torch.dtype = torch.float32,
    n_rows: int | None = None,
    num_workers: int = NUM_WORKERS,
    prefetch_factor : int = PREFETCH_FACTOR,
    pin_memory: bool = True
) -> dict[str, DataLoader]:

    base_ds = EmbeddingStreamDataset(
        path=path,
        dtype=dtype,
        n_rows=n_rows,
        batch_size=batch_size,
    )

    total_chunks = (N_ROWS if n_rows is None else n_rows + batch_size - 1) // batch_size

    train_c, val_c, test_c = split_indices(total_chunks, ratios)

    train_ds = SubsetIterable(base_ds, train_c)
    val_ds   = SubsetIterable(base_ds, val_c)
    test_ds  = SubsetIterable(base_ds, test_c)

    train_loader = DataLoader(
        train_ds,
        batch_size=1,   # Batch size is already determined by the IterableDataset.
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_chunk_collate,
        prefetch_factor=prefetch_factor
    )
    val_loader   = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_chunk_collate,
        prefetch_factor=prefetch_factor
    )
    test_loader  = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_chunk_collate,
        prefetch_factor=prefetch_factor
    )

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

def train_epoch_dl(autoencoder, loader, optimizer, loss_fn, device, logger, mean = 0, std = 1):

    autoencoder.train()
    epoch_loss, n_samples = 0.0, 0
    batch_cnt = 0

    for large_batch in loader:
        large_batch = large_batch.to(device)
        large_batch = _normalize(large_batch, mean, std)

        mini_size = BATCH_SIZE // MINIBATCH_AMOUNT

        for batch in torch.split(large_batch, mini_size, dim=0):
            
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

        if batch_cnt >= len(loader)*MINIBATCH_AMOUNT:
            break

    return epoch_loss, n_samples


def eval_epoch_dl(autoencoder, loader, loss_fn, device, mean = 0, std = 1):

    autoencoder.eval()
    epoch_loss, n_samples = 0.0, 0

    with torch.no_grad():
        batch_cnt = 0
        for batch in loader:
            batch = batch.to(device)
            x = batch
            x = _normalize(x, mean, std)
            recon = autoencoder(x)
            loss  = loss_fn(recon, x)
            epoch_loss += loss.item() * x.size(0)
            n_samples  += x.size(0)
            batch_cnt += 1

            if batch_cnt >= len(loader):
                break

    return epoch_loss, n_samples

def main():

    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    torch.set_float32_matmul_precision("high")
    
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

    best_loss = float("inf")

    loaders = create_loaders(DATASET_PATH, TRAIN_VAL_TEST_SPLIT, BATCH_SIZE, n_rows=N_ROWS)

    mean, std = _compute_stats(loaders["train"], stats_file=f"{DATASET_PATH}.stats.pt")

    mean = mean.to(device)
    std = std.to(device)

    for epoch in range(EPOCHS):
            
        epoch_loss, n_train = train_epoch_dl(autoencoder, loaders["train"],
                                                  optimizer, huber_loss, device, logger, mean=mean, std=std)
            
        if epoch > 0 and epoch % VAL_EVERY == 0:
            val_loss, n_val = eval_epoch_dl(autoencoder, loaders["val"],
                                            huber_loss, device, mean=mean, std=std)

            val_loss   = val_loss    / n_val    if n_val    else float("nan")
            if val_loss < best_loss:
                logger.info(f"Saved weights to {MODEL_FILE}")
                torch.save(autoencoder.state_dict(), MODEL_FILE)
                best_loss = val_loss

        epoch_loss = epoch_loss  / n_train if n_train  else float("nan")

        log_string = f"Epoch {epoch+1:02d} - train {epoch_loss}"

        if epoch > 0 and epoch % VAL_EVERY == 0:
            log_string += f" | val {val_loss:.8f}"

        logger.info(log_string)

        scheduler.step()

    test_loss, n_test = eval_epoch_dl(autoencoder, loaders["test"],
                                        huber_loss, device, mean=mean, std=std)

    test_loss  = test_loss   / n_test   if n_test   else float("nan")

    print(f"Test loss: {test_loss}")

if __name__ == "__main__":
    main()