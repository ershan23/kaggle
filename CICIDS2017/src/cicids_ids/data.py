"""Dataset loading, cleaning, labeling, and splitting."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


LABEL_COLUMN = "Label"
SOURCE_FILE_COLUMN = "source_file"


def find_csv_files(data_dir: str | Path) -> list[Path]:
    """Return CICIDS2017 CSV files sorted by name."""
    data_path = Path(data_dir)
    files = sorted(data_path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {data_path.resolve()}")
    return files


def normalize_label(label: object) -> str:
    """Normalize label text while preserving the attack family name."""
    return str(label).strip().replace("\ufffd", "-")


def load_cicids_csvs(
    data_dir: str | Path,
    files: Iterable[str | Path] | None = None,
    sample_size: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load CICIDS2017 CSV files into one DataFrame.

    The function strips column names, normalizes label values, and records the
    source file so file-level holdout experiments are easy to run.
    """
    paths = [Path(p) for p in files] if files is not None else find_csv_files(data_dir)
    frames: list[pd.DataFrame] = []

    for path in paths:
        df = pd.read_csv(path, low_memory=False)
        df.columns = [col.strip() for col in df.columns]
        if LABEL_COLUMN not in df.columns:
            raise ValueError(f"{path} does not contain a Label column")
        df[LABEL_COLUMN] = df[LABEL_COLUMN].map(normalize_label)
        df[SOURCE_FILE_COLUMN] = path.name
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    if sample_size is not None and 0 < sample_size < len(data):
        data = stratified_sample(data, sample_size, random_state=random_state)
    return data.reset_index(drop=True)


def stratified_sample(
    data: pd.DataFrame,
    sample_size: int,
    random_state: int = 42,
    label_col: str = LABEL_COLUMN,
) -> pd.DataFrame:
    """Take a label-stratified sample for fast experiments."""
    _, sampled = train_test_split(
        data,
        test_size=sample_size,
        stratify=data[label_col],
        random_state=random_state,
    )
    return sampled.reset_index(drop=True)


def make_target(data: pd.DataFrame, target: str = "binary") -> pd.Series:
    """Create y for binary or multiclass IDS experiments."""
    labels = data[LABEL_COLUMN].map(normalize_label)
    if target == "binary":
        return pd.Series(np.where(labels.eq("BENIGN"), 0, 1), name="target")
    if target == "multiclass":
        return labels.rename("target")
    raise ValueError("target must be 'binary' or 'multiclass'")


def get_feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Return the raw feature frame without label or source metadata."""
    drop_cols = [LABEL_COLUMN, SOURCE_FILE_COLUMN]
    return data.drop(columns=[c for c in drop_cols if c in data.columns])


def clean_numeric_features(X: pd.DataFrame) -> pd.DataFrame:
    """Convert feature columns to numeric values and replace infinities."""
    cleaned = X.apply(pd.to_numeric, errors="coerce")
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    return cleaned


def drop_duplicate_rows(
    data: pd.DataFrame,
    include_source_file: bool = False,
) -> pd.DataFrame:
    """Drop exact duplicate flow rows."""
    subset = None
    if not include_source_file and SOURCE_FILE_COLUMN in data.columns:
        subset = [c for c in data.columns if c != SOURCE_FILE_COLUMN]
    return data.drop_duplicates(subset=subset).reset_index(drop=True)


def random_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.3,
    random_state: int = 42,
):
    """Stratified random split."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def file_holdout_split(
    data: pd.DataFrame,
    holdout_files: list[str],
    target: str = "binary",
):
    """Split by source CSV file for stricter scenario generalization tests."""
    if not holdout_files:
        raise ValueError("holdout_files must contain at least one file name")
    holdout = data[SOURCE_FILE_COLUMN].isin(holdout_files)
    train_df = data.loc[~holdout].reset_index(drop=True)
    test_df = data.loc[holdout].reset_index(drop=True)
    if train_df.empty or test_df.empty:
        raise ValueError("file holdout produced an empty train or test split")

    X_train = get_feature_frame(train_df)
    y_train = make_target(train_df, target=target)
    X_test = get_feature_frame(test_df)
    y_test = make_target(test_df, target=target)
    return X_train, X_test, y_train, y_test


def dataset_profile(data: pd.DataFrame) -> dict:
    """Return EDA-ready profile statistics."""
    feature_cols = [c for c in data.columns if c not in [LABEL_COLUMN, SOURCE_FILE_COLUMN]]
    X_num = clean_numeric_features(data[feature_cols])
    raw_numeric = data[feature_cols].apply(pd.to_numeric, errors="coerce")
    inf_counts = np.isinf(raw_numeric.to_numpy(dtype=np.float64, copy=False)).sum(axis=0)

    return {
        "rows": int(len(data)),
        "features": int(len(feature_cols)),
        "source_files": data[SOURCE_FILE_COLUMN].value_counts().to_dict()
        if SOURCE_FILE_COLUMN in data.columns
        else {},
        "labels": data[LABEL_COLUMN].value_counts().to_dict(),
        "binary_labels": {
            "BENIGN": int((data[LABEL_COLUMN] == "BENIGN").sum()),
            "ATTACK": int((data[LABEL_COLUMN] != "BENIGN").sum()),
        },
        "missing_top_20": X_num.isna().sum().sort_values(ascending=False).head(20).to_dict(),
        "infinite_top_20": {
            feature_cols[i]: int(count)
            for i, count in enumerate(inf_counts)
            if int(count) > 0
        },
        "duplicate_rows_excluding_source_file": int(
            data.drop(columns=[SOURCE_FILE_COLUMN], errors="ignore").duplicated().sum()
        ),
    }
