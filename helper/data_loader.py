from pathlib import Path
import pandas as pd
import numpy as np
import torch
import jsonlines

import pyarrow.parquet as pq
            
def stream_dataset(path : str, cols : list[str], nrows : int, chunk_size : int) -> dict:
    '''
        Streams a dataset in chunks of the form {col1: [...], col2: [...] ...}.
    '''
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: {path}")

    suffix = p.suffix.lower()

    if suffix in {".json", ".jsonl", ".jsonlines"}:
        return _stream_dataset_json(path, cols, nrows, chunk_size)
    if suffix == ".parquet":
        return _stream_dataset_parquet(path, cols, nrows, chunk_size)
    else:
        raise NotImplementedError(f"File format not supported: {suffix}")

def _stream_dataset_parquet(path : str, cols : list[str], nrows : int, chunk_size : int):

    parquet_file = pq.ParquetFile(path)

    total_read = 0

    for batch in parquet_file.iter_batches(batch_size=chunk_size, columns=cols):

        data = batch.to_pydict()

        batch_len = len(data[cols[0]])

        if total_read + batch_len > nrows:
            needed = nrows - total_read
            data = {k: v[:needed] for k, v in data.items()}
            yield data
            break

        yield data
        total_read += batch_len

        if total_read >= nrows:
            break

def _stream_dataset_json(path : str, cols : list[str], nrows : int, chunk_size : int):
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