"""Unified model training entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from courtvision.data.collect import DEFAULT_SEASON
from courtvision.models import train_lgbm
from courtvision.models.train_lgbm import INNER_TRAIN_FRACTION
from courtvision.utils.config import ProjectConfig, load_project_config

SUPPORTED_MODELS: frozenset[str] = frozenset({"lightgbm"})


def processed_features_dir(config: ProjectConfig) -> Path:
    """Return the model-ready feature directory derived from project config."""
    if config.data_dir is None:
        raise SystemExit("Project config missing data_dir (required for training)")
    return config.data_dir / "processed" / "features"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CourtVision models using project configuration.",
    )
    parser.add_argument(
        "--config",
        default="configs/local.yaml",
        help="Project config YAML (default: configs/local.yaml)",
    )
    parser.add_argument(
        "--model",
        default="lightgbm",
        help="Model trainer to run (supported: lightgbm)",
    )
    parser.add_argument(
        "--mode",
        choices=("default", "search"),
        default="default",
        help="default: train once; search: hyperparameter search with inner validation",
    )
    parser.add_argument(
        "--season",
        default=DEFAULT_SEASON,
        help="Season label (e.g. 2024-25)",
    )
    parser.add_argument(
        "--inner-train-fraction",
        type=float,
        default=INNER_TRAIN_FRACTION,
        help="Fraction of earliest train games for inner train (search mode only)",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip MLflow logging",
    )
    parser.add_argument(
        "--register-candidate",
        action="store_true",
        help="Search mode only: register the best model as the candidate version",
    )
    return parser.parse_args(argv)


def _validate_cli_args(args: argparse.Namespace) -> None:
    if args.model not in SUPPORTED_MODELS:
        supported = ", ".join(sorted(SUPPORTED_MODELS))
        raise SystemExit(f"Unsupported model: {args.model}. Supported models: {supported}")
    if args.register_candidate and args.mode != "search":
        raise SystemExit("--register-candidate requires --mode search")
    if args.register_candidate and args.no_mlflow:
        raise SystemExit("--register-candidate cannot be used with --no-mlflow")


def run_from_args(args: argparse.Namespace) -> None:
    """Route CLI arguments to the selected model trainer."""
    _validate_cli_args(args)

    config = load_project_config(args.config)
    processed_dir = processed_features_dir(config)
    log_mlflow = not args.no_mlflow

    if args.model == "lightgbm":
        if args.mode == "search":
            train_lgbm.run_search(
                args.season,
                processed_dir=processed_dir,
                inner_train_fraction=args.inner_train_fraction,
                log_mlflow=log_mlflow,
                register_candidate=args.register_candidate,
                mlflow_tracking_uri=config.mlflow_tracking_uri or None,
            )
            return

        train_lgbm.run_default(
            args.season,
            processed_dir=processed_dir,
            log_mlflow=log_mlflow,
            mlflow_tracking_uri=config.mlflow_tracking_uri or None,
        )


def main() -> None:
    run_from_args(parse_args())


if __name__ == "__main__":
    main()
