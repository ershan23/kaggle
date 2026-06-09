"""Run EDA for CICIDS2017 CSV files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cicids_ids.config import dump_json, ensure_dirs, load_config
from cicids_ids.data import dataset_profile, load_cicids_csvs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.data_dir:
        config["data_dir"] = args.data_dir
    if args.sample_size is not None:
        config["sample_size"] = args.sample_size

    ensure_dirs(config["output_dir"])
    data = load_cicids_csvs(
        config["data_dir"],
        sample_size=config.get("sample_size"),
        random_state=config.get("random_state", 42),
    )
    profile = dataset_profile(data)
    output_path = Path(config["output_dir"]) / "eda_profile.json"
    dump_json(profile, output_path)
    print(f"EDA profile written to {output_path}")
    print(f"Rows: {profile['rows']:,}; features: {profile['features']}")
    print(f"Labels: {profile['labels']}")


if __name__ == "__main__":
    main()
