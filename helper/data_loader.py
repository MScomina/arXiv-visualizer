from pathlib import Path
import pandas as pd
import numpy as np
import torch
import jsonlines
import pyarrow.parquet as pq
import itertools
import math
from torch.utils.data import IterableDataset

class EmbeddingStreamDataset(IterableDataset):
    def __init__(self, path: str,
                 batch_size: int,
                 dtype: torch.dtype,
                 n_rows: int | None = None):
        super().__init__()
        self.path = path
        self.dtype = dtype
        self.n_rows = n_rows
        self.batch_size = batch_size

        total_rows = self.n_rows if self.n_rows is not None else N_ROWS
        self._num_chunks = math.ceil(total_rows / self.batch_size)

    def __iter__(self):
        for chunk in stream_dataset(
            self.path,
            ["embedding"],
            self.n_rows if self.n_rows is not None else N_ROWS,
            self.batch_size,
            needs_torch_tensor=True
        ):
            yield chunk["embedding"]

    def __len__(self):
        return self._num_chunks


class SubsetIterable(IterableDataset):

    def __init__(self, base: IterableDataset, sampler):
        super().__init__()
        self.base = base
        self.sampler = list(sampler)
        self.sampler_set = set(self.sampler)

    def __iter__(self):
        base_iter = iter(self.base)
        cur_idx = 0

        for _ in range(len(self.sampler)):
            while cur_idx not in self.sampler_set:
                next(base_iter)
                cur_idx += 1
            chunk = next(base_iter)
            cur_idx += 1
            yield chunk

    def __len__(self):
        return len(self.sampler)

def read_row_parquet(path : str, cols : list[str], predicate : list[str]) -> dict:
    row = pq.read_table(
        path,
        columns=cols,
        filters=predicate
    ).to_pydict()

    return row
    
def stream_dataset(path : str, cols : list[str], nrows : int, chunk_size : int, needs_torch_tensor : bool = False) -> dict:
    '''
        Streams a dataset in chunks of the form {col1: [...], col2: [...] ...}.
    '''
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: {path}")

    suffix = p.suffix.lower()

    if suffix in {".json", ".jsonl", ".jsonlines"}:
        return _stream_dataset_jsonl(path, cols, nrows, chunk_size)
    if suffix == ".parquet":
        return _stream_dataset_parquet(path, cols, nrows, chunk_size, needs_torch_tensor=needs_torch_tensor)
    else:
        raise NotImplementedError(f"File format not supported: {suffix}")

def _stream_dataset_parquet(path : str, cols : list[str], nrows : int, chunk_size : int, needs_torch_tensor : bool = False):

    parquet_file = pq.ParquetFile(path, buffer_size=1024*1024, memory_map=True, pre_buffer=True)

    total_read = 0

    for batch in parquet_file.iter_batches(batch_size=chunk_size, columns=cols, use_threads=True):
        batch_len = len(batch)
        if needs_torch_tensor:
            yield {k: torch.Tensor(v).to(torch.float32) for k, v in batch.to_pydict().items()}
        else:
            data = batch.to_pydict()

            if total_read + batch_len > nrows:
                needed = nrows - total_read
                data = {k: v[:needed] for k, v in data.items()}
                yield data
                break

            yield data
        total_read += batch_len

        if total_read >= nrows:
            break

    parquet_file.close()

def _stream_dataset_jsonl(path : str, cols : list[str], nrows : int, chunk_size : int):
    current_line = 0
    with jsonlines.open(path) as reader:
        partial_data = {k : [] for k in cols}
        for obj in reader:
            for k in cols:
                partial_data[k].append(obj[k])
            current_line += 1
            if current_line >= nrows:
                if len(partial_data[cols[0]]) != 0:
                    yield partial_data
                    partial_data = {k : [] for k in cols}
                break
            if len(partial_data[cols[0]]) >= chunk_size:
                yield partial_data
                partial_data = {k : [] for k in cols}
        
        if len(partial_data[cols[0]]) != 0:
            yield partial_data