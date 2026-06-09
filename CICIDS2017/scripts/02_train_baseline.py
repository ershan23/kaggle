"""Train RandomForest, XGBoost, and LightGBM baselines."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cicids_ids.config import dump_json, ensure_dirs, load_config
from cicids_ids.metrics import binary_metrics, metrics_table
from cicids_ids.models import make_pipeline, model_specs
from cicids_ids.workflow import load_xy_from_config, save_model_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--model", action="append", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.sample_size is not None:
        config["sample_size"] = args.sample_size
    selected_models = args.model or config.get("models", [])

    ensure_dirs(config["output_dir"], config["model_dir"])
    X_train, X_test, y_train, y_test = load_xy_from_config(config)
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    specs = model_specs(
        random_state=config.get("random_state", 42),
        scale_pos_weight=neg / max(pos, 1),
    )

    records = []
    for model_name in selected_models:
        if model_name not in specs:
            print(f"Skipping unavailable model: {model_name}")
            continue
        spec = specs[model_name]
        print(f"Training {model_name} ...")
        started = time.time()
        pipeline = make_pipeline(spec.estimator)
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        record = {"model": model_name, **binary_metrics(y_test, pred)}
        record["fit_predict_seconds"] = round(time.time() - started, 2)
        records.append(record)

        save_model_bundle(
            pipeline,
            Path(config["model_dir"]) / f"{model_name}.joblib",
            feature_columns=list(X_train.columns),
            metadata={"config": config, "metrics": record},
        )

    table = metrics_table(records)
    table_path = Path(config["output_dir"]) / "baseline_metrics.csv"
    json_path = Path(config["output_dir"]) / "baseline_metrics.json"
    table.to_csv(table_path, index=False)
    dump_json(records, json_path)
    print(table.to_string(index=False))
    print(f"Metrics written to {table_path} and {json_path}")


if __name__ == "__main__":
    main()
