"""Domain feature engineering for flow-based IDS data."""

from __future__ import annotations

import numpy as np
import pandas as pd


ZERO_VARIANCE_COLUMNS = [
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def add_domain_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable flow behavior features.

    These features make ratios explicit for models and help interviews because
    they map back to network behavior: directionality, flow intensity, and TCP
    flag activity.
    """
    X = X.copy()

    def has(*cols: str) -> bool:
        return all(col in X.columns for col in cols)

    if has("Total Fwd Packets", "Total Backward Packets"):
        X["Total Packets"] = X["Total Fwd Packets"] + X["Total Backward Packets"]
        X["Fwd Bwd Packet Ratio"] = _safe_divide(
            X["Total Fwd Packets"], X["Total Backward Packets"] + 1
        )

    if has("Total Length of Fwd Packets", "Total Length of Bwd Packets"):
        X["Total Bytes"] = X["Total Length of Fwd Packets"] + X["Total Length of Bwd Packets"]
        X["Fwd Bwd Byte Ratio"] = _safe_divide(
            X["Total Length of Fwd Packets"], X["Total Length of Bwd Packets"] + 1
        )

    if "Total Bytes" in X.columns and "Total Packets" in X.columns:
        X["Bytes Per Packet"] = _safe_divide(X["Total Bytes"], X["Total Packets"])

    if has("Flow Duration", "Total Packets"):
        X["Packets Per Duration"] = _safe_divide(X["Total Packets"], X["Flow Duration"] + 1)

    if has("Active Mean", "Idle Mean"):
        X["Active Idle Ratio"] = _safe_divide(X["Active Mean"], X["Idle Mean"] + 1)

    flag_cols = [
        "FIN Flag Count",
        "SYN Flag Count",
        "RST Flag Count",
        "PSH Flag Count",
        "ACK Flag Count",
        "URG Flag Count",
        "CWE Flag Count",
        "ECE Flag Count",
    ]
    present_flags = [col for col in flag_cols if col in X.columns]
    if present_flags:
        X["TCP Flag Count Sum"] = X[present_flags].sum(axis=1)

    return X.replace([np.inf, -np.inf], np.nan)


def drop_low_information_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Drop known zero-variance columns when present."""
    return X.drop(columns=[c for c in ZERO_VARIANCE_COLUMNS if c in X.columns], errors="ignore")
