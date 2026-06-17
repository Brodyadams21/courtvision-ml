"""One-command local ML pipeline orchestrator."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from courtvision.data.collect import DEFAULT_SEASON
from courtvision.utils.config import ProjectConfig, load_project_config

RunCommand = Callable[[Sequence[str]], subprocess.CompletedProcess[bytes]]


def processed_features_dir(config: ProjectConfig) -> Path:
    """Return the model-ready feature directory derived from project config."""
    if config.data_dir is None:
        raise SystemExit("Project config missing data_dir (required for pipeline)")
    return config.data_dir / "processed" / "features"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CourtVision local ML pipeline (features optional, then train).",
    )
    parser.add_argument(
        "--config",
        default="configs/local.yaml",
        help="Project config YAML (default: configs/local.yaml)",
    )
    parser.add_argument(
        "--season",
        default=DEFAULT_SEASON,
        help="Season label (e.g. 2024-25)",
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
        "--no-mlflow",
        action="store_true",
        help="Skip MLflow logging during training",
    )
    parser.add_argument(
        "--register-candidate",
        action="store_true",
        help="Search mode only: register the best model as the candidate version",
    )
    parser.add_argument(
        "--rebuild-features",
        action="store_true",
        help=(
            "Export train/test Parquet from existing gold features before training "
            "(does not collect NBA data or reload PostgreSQL)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run without executing them",
    )
    return parser.parse_args(argv)


def _validate_cli_args(args: argparse.Namespace) -> None:
    if args.register_candidate and args.mode != "search":
        raise SystemExit("--register-candidate requires --mode search")
    if args.register_candidate and args.no_mlflow:
        raise SystemExit("--register-candidate cannot be used with --no-mlflow")


def plan_feature_export_command(season: str, config: ProjectConfig) -> list[str]:
    """Export train/test Parquet from existing gold features (no DB reload)."""
    processed_dir = processed_features_dir(config)
    return [
        sys.executable,
        "-m",
        "courtvision.data.build_features",
        "--season",
        season,
        "--export-only",
        "--processed-dir",
        str(processed_dir),
    ]


def plan_train_command(args: argparse.Namespace) -> list[str]:
    """Build the unified training entrypoint command."""
    command = [
        sys.executable,
        "-m",
        "courtvision.models.train",
        "--config",
        args.config,
        "--model",
        args.model,
        "--mode",
        args.mode,
        "--season",
        args.season,
    ]
    if args.no_mlflow:
        command.append("--no-mlflow")
    if args.register_candidate:
        command.append("--register-candidate")
    return command


def plan_pipeline_commands(
    args: argparse.Namespace,
    config: ProjectConfig,
) -> list[list[str]]:
    """Return ordered subprocess commands for the selected pipeline stages."""
    commands: list[list[str]] = []
    if args.rebuild_features:
        commands.append(plan_feature_export_command(args.season, config))
    commands.append(plan_train_command(args))
    return commands


def format_command(command: Sequence[str]) -> str:
    """Format a command for human-readable output."""
    display = list(command)
    if display and Path(display[0]).name.lower().startswith("python"):
        display[0] = "python"
    return " ".join(display)


def run_command(
    command: Sequence[str],
    *,
    dry_run: bool,
    run_command_fn: RunCommand,
) -> None:
    """Execute one pipeline stage, or print it when ``dry_run`` is set."""
    printable = format_command(command)
    if dry_run:
        print(printable)
        return

    result = run_command_fn(command)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def run_from_args(
    args: argparse.Namespace,
    *,
    run_command_fn: RunCommand | None = None,
) -> None:
    """Load config and run (or preview) the local pipeline stages."""
    _validate_cli_args(args)
    config = load_project_config(args.config)
    runner = run_command_fn or (lambda cmd: subprocess.run(cmd, check=False))
    commands = plan_pipeline_commands(args, config)

    if args.dry_run:
        print("Would run:")

    for command in commands:
        run_command(command, dry_run=args.dry_run, run_command_fn=runner)


def main() -> None:
    run_from_args(parse_args())


if __name__ == "__main__":
    main()
