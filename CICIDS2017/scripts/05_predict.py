"""Run inference on a CICIDS2017-style CSV file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cicids_ids.config import ensure_dirs
from cicids_ids.data import get_feature_frame, normalize_label
from cicids_ids.workflow import build_features, load_model_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", default="outputs/predictions.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_model_bundle(args.model_path)
    df = pd.read_csv(args.input_csv, low_memory=False)
    df.columns = [col.strip() for col in df.columns]
    if "Label" in df.columns:
        df["Label"] = df["Label"].map(normalize_label)

    X = get_feature_frame(df)
    X = build_features(X, feature_engineering=True, drop_zero_variance=True)
    X = X.reindex(columns=bundle["feature_columns"])

    model = bundle["model"]
    pred = model.predict(X)
    output = df.copy()
    output["predicted_target"] = pred
    if hasattr(model, "predict_proba"):
        output["attack_probability"] = model.predict_proba(X)[:, 1]

    output_path = Path(args.output_csv)
    ensure_dirs(output_path.parent)
    output.to_csv(output_path, index=False)
    print(f"Predictions written to {output_path}")


if __name__ == "__main__":
    main()
