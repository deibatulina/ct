#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path

import yaml

import config
from src.training.train import (
    run_baseline_experiment,
    run_linear_aligner_experiment,
    run_mlp_lora_experiment,
)
from src.utils.config import load_yaml_config


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run baseline, linear_aligner, and mlp_lora smoke tests on a tiny subset.",
    )
    parser.add_argument(
        "--records",
        type=int,
        default=3,
        help="How many records to use from each split (train/val/test). Default: 3.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="How many epochs to run in the smoke test. Default: 1.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch size override. Defaults to min(original_batch_size, records).",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional root directory for smoke test outputs.",
    )
    return parser


def resolve_output_root(custom_root: str | None) -> Path:
    if custom_root is not None:
        path = Path(custom_root)
        if not path.is_absolute():
            path = config.PROJECT_ROOT / path
        return path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return config.PROJECT_ROOT / "outputs" / "smoke_tests" / timestamp


def build_smoke_config(base_config_path: str, output_root: Path, records: int, epochs: int, batch_size: int | None):
    payload = copy.deepcopy(load_yaml_config(base_config_path))

    payload.setdefault("runtime", {})
    payload["runtime"]["max_train_records"] = records
    payload["runtime"]["max_val_records"] = records
    payload["runtime"]["max_test_records"] = records

    payload["training"]["num_epochs"] = epochs
    if batch_size is None:
        payload["training"]["batch_size"] = max(1, min(payload["training"]["batch_size"], records))
    else:
        payload["training"]["batch_size"] = batch_size

    payload["outputs"]["root_dir"] = str(output_root)
    return payload


def write_yaml(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def verify_artifacts(summary):
    artifacts = summary["artifacts"]
    missing = []

    for key, raw_path in artifacts.items():
        path = Path(raw_path)
        if key == "root_dir":
            exists = path.is_dir()
        else:
            exists = path.is_file()

        if not exists:
            missing.append(f"{key}: {path}")

    if missing:
        raise RuntimeError("Smoke test finished but some artifacts are missing:\n" + "\n".join(missing))

    summary_path = Path(artifacts["summary"])
    metrics_path = Path(artifacts["metrics"])
    epoch_logs_path = Path(artifacts["epoch_logs"])

    json.loads(summary_path.read_text(encoding="utf-8"))
    json.loads(metrics_path.read_text(encoding="utf-8"))
    epoch_logs = json.loads(epoch_logs_path.read_text(encoding="utf-8"))

    if not isinstance(epoch_logs, list) or not epoch_logs:
        raise RuntimeError(f"Epoch logs are empty or malformed: {epoch_logs_path}")


def run_smoke_experiment(name: str, base_config_path: str, runner, run_root: Path, records: int, epochs: int, batch_size: int | None):
    experiment_output_root = run_root / name
    generated_config_path = run_root / "configs" / f"{name}.smoke.yaml"

    smoke_config = build_smoke_config(
        base_config_path=base_config_path,
        output_root=experiment_output_root,
        records=records,
        epochs=epochs,
        batch_size=batch_size,
    )
    write_yaml(generated_config_path, smoke_config)

    print(f"[{name}] config: {generated_config_path}")
    print(f"[{name}] outputs: {experiment_output_root}")
    summary = runner(config_path=str(generated_config_path))
    verify_artifacts(summary)
    print(f"[{name}] ok")
    return summary


def main():
    args = build_parser().parse_args()

    if args.records <= 0:
        raise ValueError("--records must be greater than 0")
    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than 0")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")

    run_root = resolve_output_root(args.output_root)
    run_root.mkdir(parents=True, exist_ok=True)

    print(f"Smoke test run root: {run_root}")
    print(f"Records per split: {args.records}")
    print(f"Epochs: {args.epochs}")

    baseline_summary = run_smoke_experiment(
        name="baseline",
        base_config_path="configs/baseline.yaml",
        runner=run_baseline_experiment,
        run_root=run_root,
        records=args.records,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    aligner_summary = run_smoke_experiment(
        name="linear_aligner",
        base_config_path="configs/linear_aligner.yaml",
        runner=run_linear_aligner_experiment,
        run_root=run_root,
        records=args.records,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    mlp_lora_summary = run_smoke_experiment(
        name="mlp_lora",
        base_config_path="configs/mlp_lora.yaml",
        runner=run_mlp_lora_experiment,
        run_root=run_root,
        records=args.records,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    result = {
        "status": "ok",
        "run_root": str(run_root),
        "records_per_split": args.records,
        "epochs": args.epochs,
        "baseline_summary_path": baseline_summary["artifacts"]["summary"],
        "linear_aligner_summary_path": aligner_summary["artifacts"]["summary"],
        "mlp_lora_summary_path": mlp_lora_summary["artifacts"]["summary"],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
