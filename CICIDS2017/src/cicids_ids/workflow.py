"""Shared workflow utilities for training scripts."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from cicids_ids.data import (
    clean_numeric_features,
    drop_duplicate_rows,
    file_holdout_split,
    get_feature_frame,
    load_cicids_csvs,
    make_target,
    random_train_test_split,
)
from cicids_ids.features import add_domain_features, drop_low_information_columns


def build_features(
    X: pd.DataFrame,
    feature_engineering: bool = True,
    drop_zero_variance: bool = True,
) -> pd.DataFrame:
    """Apply numeric cleaning and optional feature engineering."""
    X = clean_numeric_features(X)
    if feature_engineering:
        X = add_domain_features(X)
    if drop_zero_variance:
        X = drop_low_information_columns(X)
    return X


def load_xy_from_config(config: dict):
    """Load data, apply optional de-duplication, split, and engineer features."""
    data = load_cicids_csvs(
        config["data_dir"],
        sample_size=config.get("sample_size"),
        random_state=config.get("random_state", 42),
    )
    if config.get("deduplicate", False):
        data = drop_duplicate_rows(data)

    if config.get("split_strategy", "random") == "file_holdout":
        X_train, X_test, y_train, y_test = file_holdout_split(
            data,
            holdout_files=config.get("holdout_files", []),
            target=config.get("target", "binary"),
        )
    else:
        X = get_feature_frame(data)
        y = make_target(data, target=config.get("target", "binary"))
        X_train, X_test, y_train, y_test = random_train_test_split(
            X,
            y,
            test_size=config.get("test_size", 0.3),
            random_state=config.get("random_state", 42),
        )

    X_train = build_features(
        X_train,
        feature_engineering=config.get("feature_engineering", True),
        drop_zero_variance=config.get("drop_zero_variance", True),
    )
    X_test = build_features(
        X_test,
        feature_engineering=config.get("feature_engineering", True),
        drop_zero_variance=config.get("drop_zero_variance", True),
    )
    X_test = X_test.reindex(columns=X_train.columns)
    return X_train, X_test, y_train, y_test


def save_model_bundle(
    model,
    path: str | Path,
    feature_columns: list[str],
    metadata: dict,
) -> None:
    """Save a fitted model with the feature schema and metadata."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "metadata": metadata,
        },
        output,
    )


def load_model_bundle(path: str | Path) -> dict:
    """Load a model bundle saved by save_model_bundle."""
    return joblib.load(path)
