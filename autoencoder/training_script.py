import numpy as np
import torch
import hashlib
import pandas as pd
from autoencoder import VAE

EPOCHS = 50
BATCH_SIZE = 4096
TRAINING_RATE = 0.0005
GAMMA = 0.95
DROPOUT = 0.1
WEIGHT_DECAY = 1e-5
RANDOM_STATE = 31412718

IN_DIMENSIONS = 768
HIDDEN_LAYERS = (512, 384, 256, 128)
LATENT_SPACE = 64
CHUNKSIZE = BATCH_SIZE*16

DATASET_PATH = "./data/processed/embeddings.json"

PRINT_EVERY = 25

def _normalize(batch : torch.Tensor) -> torch.Tensor:

    batch = (batch - batch.mean(dim=1, keepdim=True)) / batch.std(dim=1, keepdim=True).clamp_min(1e-8)
    return batch / batch.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)

def load_embeddings_lazy(path: str, batch_size: int = 4096, dtype=torch.float32, ratios : tuple[float, float, float] = (0.7, 0.15, 0.15)):
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("The split ratios must sum to 1.0")
    thresholds = np.cumsum(np.array(ratios))

    def _bucket(idx: int) -> str:
        rnd_bytes = hashlib.sha256(str(idx).encode()).digest()
        rnd_val = int.from_bytes(rnd_bytes[:8], "big") / 2**64

        if rnd_val < thresholds[0]:
            return "train"
        elif rnd_val < thresholds[1]:
            return "val"
        else:
            return "test"

    buffers = {"train": [], "val": [], "test": []}
    idx = 0
    for chunk in pd.read_json(path, lines=True, chunksize=CHUNKSIZE):
        for emb in chunk["embedding"]:
            bucket = _bucket(idx)
            buffers[bucket].append(emb)
            idx += 1

            for bname in ("train", "val", "test"):
                if len(buffers[bname]) >= batch_size:
                    batch = buffers[bname][:batch_size]
                    buffers[bname] = buffers[bname][batch_size:]
                    yield (bname, torch.tensor(batch, dtype=dtype))

    for bname in ("train", "val", "test"):
        if buffers[bname]:
            yield (bname, torch.tensor(buffers[bname], dtype=dtype))

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    autoencoder = VAE(
        dropout=DROPOUT,
        in_dimensions=IN_DIMENSIONS,
        hidden_layers=HIDDEN_LAYERS,
        latent_size=LATENT_SPACE,
        beta=0.0
    ).to(device=device)

    optimizer = torch.optim.AdamW(
        autoencoder.parameters(),
        lr=TRAINING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=5, 
        T_mult=1, 
        eta_min=1e-6
    )
    criterion = torch.nn.MSELoss(reduction="mean")

    for epoch in range(EPOCHS):
        autoencoder.train()
        autoencoder.var_encoder.beta = min(1.0, (((2*epoch)+1)/(EPOCHS)))

        epoch_loss, n_train   = 0.0, 0
        val_loss,  n_val      = 0.0, 0
        test_loss, n_test     = 0.0, 0
        train_batch_cnt = 0

        for tag, batch in load_embeddings_lazy(
                DATASET_PATH,
                batch_size=BATCH_SIZE,
                dtype=torch.float32,
                ratios=(0.7, 0.15, 0.15)):
            x = batch.to(device)
            x = _normalize(batch.to(device))

            if tag == "train":
                optimizer.zero_grad()
                recon, mu, logv, kl = autoencoder(x)
                loss  = criterion(recon, x) + kl
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * x.size(0)
                n_train   += x.size(0)
                train_batch_cnt += 1

                if train_batch_cnt % PRINT_EVERY == 0:
                    avg_loss = epoch_loss / n_train
                    print(f"[{train_batch_cnt:>6d} train] "
                        f"avg train loss (so far) = {avg_loss:.8f}")

            elif tag == "val":
                with torch.no_grad():
                    recon, mu, logv, kl = autoencoder(x)
                    loss  = criterion(recon, x) + kl
                val_loss += loss.item() * x.size(0)
                n_val    += x.size(0)

            else:
                with torch.no_grad():
                    recon, mu, logv, kl = autoencoder(x)
                    loss  = criterion(recon, x) + kl
                test_loss += loss.item() * x.size(0)
                n_test    += x.size(0)

        epoch_loss = epoch_loss  / n_train if n_train  else float("nan")
        val_loss   = val_loss    / n_val    if n_val    else float("nan")
        test_loss  = test_loss   / n_test   if n_test   else float("nan")

        print(f"Epoch {epoch+1:02d} – "
            f"train {epoch_loss:.8f} | "
            f"val   {val_loss:.8f} | "
            f"test  {test_loss:.8f}")

        scheduler.step()

    torch.save(autoencoder.var_encoder.state_dict(), f"./models/ae_encoder - {HIDDEN_LAYERS} - {LATENT_SPACE}.pt")
    print(f"Saved encoder weights to ./models/ae_encoder - {HIDDEN_LAYERS} - {LATENT_SPACE}.pt")


if __name__ == "__main__":
    main()