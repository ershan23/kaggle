"""Tune tree models with RandomizedSearchCV."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

from sklearn.model_selection import RandomizedSearchCV

from cicids_ids.config import dump_json, ensure_dirs, load_config
from cicids_ids.metrics import binary_metrics, metrics_table
from cicids_ids.models import make_pipeline, model_specs
from cicids_ids.workflow import load_xy_from_config, save_model_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--model", action="append", default=None)
    parser.add_argument("--n-iter", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.sample_size is not None:
        config["sample_size"] = args.sample_size
    if args.n_iter is not None:
        config["tuning"]["n_iter"] = args.n_iter
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
        print(f"Tuning {model_name} ...")
        search = RandomizedSearchCV(
            estimator=make_pipeline(spec.estimator),
            param_distributions=spec.param_distributions,
            n_iter=config["tuning"].get("n_iter", 12),
            scoring=config["tuning"].get("scoring", "f1"),
            cv=config["tuning"].get("cv", 3),
            n_jobs=config["tuning"].get("n_jobs", -1),
            random_state=config.get("random_state", 42),
            verbose=1,
        )
        search.fit(X_train, y_train)
        pred = search.best_estimator_.predict(X_test)
        record = {
            "model": model_name,
            "best_score_cv": float(search.best_score_),
            "best_params": search.best_params_,
            **binary_metrics(y_test, pred),
        }
        records.append(record)
        save_model_bundle(
            search.best_estimator_,
            Path(config["model_dir"]) / f"{model_name}_tuned.joblib",
            feature_columns=list(X_train.columns),
            metadata={"config": config, "metrics": record},
        )

    table = metrics_table(records)
    table_path = Path(config["output_dir"]) / "tuned_metrics.csv"
    json_path = Path(config["output_dir"]) / "tuned_metrics.json"
    table.drop(columns=["best_params"], errors="ignore").to_csv(table_path, index=False)
    dump_json(records, json_path)
    print(table.drop(columns=["best_params"], errors="ignore").to_string(index=False))
    print(f"Tuning outputs written to {table_path} and {json_path}")


if __name__ == "__main__":
    main()
