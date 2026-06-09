"""Train a soft-voting or stacking ensemble from configured base models."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

from cicids_ids.config import dump_json, ensure_dirs, load_config
from cicids_ids.metrics import binary_metrics
from cicids_ids.models import (
    make_pipeline,
    make_soft_voting_ensemble,
    make_stacking_ensemble,
    model_specs,
)
from cicids_ids.workflow import load_xy_from_config, save_model_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--method", choices=["soft_voting", "stacking"], default="soft_voting")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.sample_size is not None:
        config["sample_size"] = args.sample_size

    ensure_dirs(config["output_dir"], config["model_dir"])
    X_train, X_test, y_train, y_test = load_xy_from_config(config)
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    specs = model_specs(
        random_state=config.get("random_state", 42),
        scale_pos_weight=neg / max(pos, 1),
    )

    estimators = []
    for model_name in config.get("models", []):
        if model_name in specs:
            estimators.append((model_name, make_pipeline(specs[model_name].estimator)))

    if len(estimators) < 2:
        raise ValueError("Need at least two available models for an ensemble")

    if args.method == "stacking":
        ensemble = make_stacking_ensemble(
            estimators,
            cv=config["tuning"].get("cv", 3),
            random_state=config.get("random_state", 42),
        )
    else:
        ensemble = make_soft_voting_ensemble(estimators)
    ensemble.fit(X_train, y_train)
    pred = ensemble.predict(X_test)
    record = {"model": f"{args.method}_ensemble", **binary_metrics(y_test, pred)}

    save_model_bundle(
        ensemble,
        Path(config["model_dir"]) / f"{args.method}_ensemble.joblib",
        feature_columns=list(X_train.columns),
        metadata={"config": config, "metrics": record},
    )
    dump_json(record, Path(config["output_dir"]) / f"{args.method}_ensemble_metrics.json")
    print(record)


if __name__ == "__main__":
    main()
