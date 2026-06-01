import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import OOF_DIR, PREDS_DIR


def get_logger(name: str = "stellar") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == object or pd.api.types.is_string_dtype(col_type):
            continue
        if pd.api.types.is_float_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif pd.api.types.is_integer_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="integer")
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        logger = get_logger()
        logger.info(f"Memory: {start_mem:.1f}MB -> {end_mem:.1f}MB ({100*(start_mem-end_mem)/start_mem:.1f}% reduction)")
    return df


def save_array(arr: np.ndarray, name: str, subdir: str = "oof") -> Path:
    d = OOF_DIR if subdir == "oof" else PREDS_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.npy"
    np.save(path, arr)
    return path


def load_array(name: str, subdir: str = "oof") -> np.ndarray:
    d = OOF_DIR if subdir == "oof" else PREDS_DIR
    return np.load(d / f"{name}.npy")


class Timer:
    def __init__(self, msg: str = ""):
        self.msg = msg
        self.logger = get_logger()

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        self.logger.info(f"{self.msg} done in {elapsed:.1f}s")
