from pathlib import Path
import pandas as pd
import numpy as np
import torch
import json
import jsonlines
import pyarrow.parquet as pq
import sqlite3
import math
import re
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
    
def fetch_rows(conn : sqlite3.Connection, search_substring : str, target_col : str, select_cols : list[str], max_rows : int = 10, table_name : str = "data"):

    quoted_cols = ", ".join(f'"{c}"' for c in select_cols)
    quoted_target_col = f'"{target_col}"'
    quoted_table = f'"{table_name}"'

    sql = f"""
        SELECT {quoted_cols}
        FROM {quoted_table}
        WHERE {quoted_target_col} LIKE ?
        ORDER BY id
        LIMIT ?
    """

    pattern = f"%{search_substring}%"

    cursor = conn.cursor()
    cursor.execute(sql, (pattern, max_rows))
    result = cursor.fetchall()

    col_names = [desc[0] for desc in cursor.description]
    data_dict: dict[str, list] = {col: [] for col in col_names}
    for row in result:
        for col, val in zip(col_names, row):
            data_dict[col].append(val)

    return data_dict
        
def stream_dataset(path : str, cols : list[str], n_rows : int, chunk_size : int, needs_torch_tensor : bool = False) -> dict:
    '''
        Streams a dataset in chunks of the form {col1: [...], col2: [...] ...}.
    '''
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: {path}")

    suffix = p.suffix.lower()

    if suffix in {".json", ".jsonl", ".jsonlines"}:
        return _stream_dataset_jsonl(path, cols, n_rows, chunk_size)
    if suffix == ".parquet":
        return _stream_dataset_parquet(path, cols, n_rows, chunk_size, needs_torch_tensor=needs_torch_tensor)
    if suffix in {".sqlite3", ".sql"}:
        return _stream_dataset_sql(path, cols, n_rows, chunk_size)
    else:
        raise NotImplementedError(f"File format not supported: {suffix}")

def _stream_dataset_sql(path : str, cols : list[str], n_rows : int, chunk_size : int):
    
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
    table = cursor.fetchone()
    if not table:
        conn.close()
        return
    table_name = table[0]
    cols_sql = ", ".join([f'"{c}"' for c in cols])
    query = f"SELECT {cols_sql} FROM {table_name} ORDER BY id"
    cursor.execute(query)
    total_read = 0
    while total_read < n_rows:
        rows = cursor.fetchmany(chunk_size)
        if not rows:
            break
        batch_len = len(rows)
        data = {col: [] for col in cols}
        for row in rows:
            for idx, col in enumerate(cols):
                if col == "embedding":
                    data[col].append(json.loads(row[idx]))
                else:
                    data[col].append(row[idx])
        needed = min(batch_len, n_rows - total_read)
        if needed < batch_len:
            for col in cols:
                data[col] = data[col][:needed]
            yield data
            break
        yield data
        total_read += batch_len
    conn.close()

def _stream_dataset_parquet(path : str, cols : list[str], n_rows : int, chunk_size : int, needs_torch_tensor : bool = False):

    parquet_file = pq.ParquetFile(path, buffer_size=1024*1024, memory_map=True, pre_buffer=True)

    total_read = 0

    for batch in parquet_file.iter_batches(batch_size=chunk_size, columns=cols, use_threads=True):
        batch_len = len(batch)
        if needs_torch_tensor:
            yield {k: torch.Tensor(v).to(torch.float32) for k, v in batch.to_pydict().items()}
        else:
            data = batch.to_pydict()

            if total_read + batch_len > n_rows:
                needed = n_rows - total_read
                data = {k: v[:needed] for k, v in data.items()}
                yield data
                break

            yield data
        total_read += batch_len

        if total_read >= n_rows:
            break

    parquet_file.close()

def _stream_dataset_jsonl(path : str, cols : list[str], n_rows : int, chunk_size : int):
    current_line = 0
    with jsonlines.open(path) as reader:
        partial_data = {k : [] for k in cols}
        for obj in reader:
            for k in cols:
                partial_data[k].append(obj[k])
            current_line += 1
            if current_line >= n_rows:
                if len(partial_data[cols[0]]) != 0:
                    yield partial_data
                    partial_data = {k : [] for k in cols}
                break
            if len(partial_data[cols[0]]) >= chunk_size:
                yield partial_data
                partial_data = {k : [] for k in cols}
        
        if len(partial_data[cols[0]]) != 0:
            yield partial_data