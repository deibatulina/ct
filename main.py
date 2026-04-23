import argparse
import json
import sys

import config
from src.data.preprocessing import preprocess_dataset
from src.utils.summary import format_preprocessing_summary


def build_parser():
    parser = argparse.ArgumentParser(
        description="Entry point for CT-scans experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Load, normalize, and split the configured dataset.",
    )
    preprocess_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for the number of streamed raw rows to preprocess.",
    )
    preprocess_parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved configuration before execution.",
    )
    preprocess_parser.add_argument(
        "--download-batch-size",
        type=int,
        default=None,
        help="How many NIfTI volumes to download and preprocess per chunk.",
    )

    baseline_parser = subparsers.add_parser(
        "baseline",
        help="Run the baseline CT-to-text experiment.",
    )
    baseline_parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="Path to the baseline YAML config.",
    )
    baseline_parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved baseline config before execution.",
    )

    linear_aligner_parser = subparsers.add_parser(
        "linear_aligner",
        help="Run the linear_aligner CT-to-text experiment.",
    )
    linear_aligner_parser.add_argument(
        "--config",
        default="configs/linear_aligner.yaml",
        help="Path to the linear_aligner YAML config.",
    )
    linear_aligner_parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved linear_aligner config before execution.",
    )

    mlp_lora_parser = subparsers.add_parser(
        "mlp_lora",
        help="Run the mlp_lora CT-to-text experiment.",
    )
    mlp_lora_parser.add_argument(
        "--config",
        default="configs/mlp_lora.yaml",
        help="Path to the mlp_lora YAML config.",
    )
    mlp_lora_parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved mlp_lora config before execution.",
    )

    return parser


def handle_preprocess(limit=None, download_batch_size=None):
    summary = preprocess_dataset(limit=limit, download_batch_size=download_batch_size)
    print(format_preprocessing_summary(summary))


def handle_baseline(config_path):
    from src.training.train import run_baseline_experiment

    summary = run_baseline_experiment(config_path=config_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def handle_linear_aligner(config_path):
    from src.training.train import run_linear_aligner_experiment

    summary = run_linear_aligner_experiment(config_path=config_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def handle_mlp_lora(config_path):
    from src.training.train import run_mlp_lora_experiment

    summary = run_mlp_lora_experiment(config_path=config_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    parser = build_parser()
    try:
        args = parser.parse_args()

        if args.command == "preprocess" and getattr(args, "print_config", False):
            print(json.dumps(config.to_printable(), indent=2, ensure_ascii=False))

        if args.command == "preprocess":
            handle_preprocess(limit=args.limit, download_batch_size=args.download_batch_size)
            return

        if args.command == "baseline":
            if args.print_config:
                from src.utils.config import load_yaml_config

                print(json.dumps(load_yaml_config(args.config), indent=2, ensure_ascii=False))
            handle_baseline(config_path=args.config)
            return

        if args.command == "linear_aligner":
            if args.print_config:
                from src.utils.config import load_yaml_config

                print(json.dumps(load_yaml_config(args.config), indent=2, ensure_ascii=False))
            handle_linear_aligner(config_path=args.config)
            return

        if args.command == "mlp_lora":
            if args.print_config:
                from src.utils.config import load_yaml_config

                print(json.dumps(load_yaml_config(args.config), indent=2, ensure_ascii=False))
            handle_mlp_lora(config_path=args.config)
            return

        parser.error(f"Unsupported command: {args.command}")
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
