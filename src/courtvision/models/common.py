"""Shared data loading, evaluation, and plotting for shot-make model training."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from courtvision.data.build_features import processed_train_test_paths
from courtvision.data.collect import DEFAULT_SEASON
from courtvision.data.load_data import PROJECT_ROOT

DEFAULT_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
CALIBRATION_N_BINS = 10

TARGET_COLUMN = "shot_made_flag"

FEATURE_COLUMNS = [
    "shot_value",
    "shot_distance",
    "loc_x",
    "loc_y",
    "abs_loc_x",
    "shot_angle",
    "is_corner_three",
    "is_home",
    "period",
    "seconds_remaining_period",
    "seconds_remaining_game",
    "score_margin",
    "score_margin_missing",
    "player_recent_fg_pct_5",
    "player_recent_fg3_pct_5",
    "player_recent_fga_5",
    "player_recent_fg3a_5",
    "player_recent_minutes_5",
    "player_recent_points_5",
    "team_recent_off_eff_proxy_5",
    "team_recent_pace_proxy_5",
    "team_recent_fg_pct_5",
    "team_recent_three_point_rate_5",
    "team_recent_fga_5",
    "team_recent_points_5",
    "team_recent_turnovers_5",
    "opp_recent_points_allowed_5",
    "opp_recent_fg_pct_allowed_5",
    "opp_recent_three_point_rate_allowed_5",
    "opp_recent_pace_proxy_5",
    "opp_recent_fga_allowed_5",
]

SCORE_MARGIN_FEATURE_COLUMNS: tuple[str, ...] = (
    "score_margin",
    "score_margin_missing",
)

# Always audit score-margin columns even if FEATURE_COLUMNS changes.
LEAKAGE_AUDIT_FEATURES: tuple[str, ...] = tuple(
    dict.fromkeys([*FEATURE_COLUMNS, *SCORE_MARGIN_FEATURE_COLUMNS])
)
SUSPICIOUS_AUC_THRESHOLD = 0.90


def load_train_test_parquet(
    season: str = DEFAULT_SEASON,
    *,
    processed_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test shot-feature Parquet files for a season."""
    paths = processed_train_test_paths(season, output_dir=processed_dir)
    train = pd.read_parquet(paths["train"])
    test = pd.read_parquet(paths["test"])
    return train, test


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return numeric/boolean feature matrix X and binary target y."""
    required = [*FEATURE_COLUMNS, TARGET_COLUMN]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise KeyError(f"Frame missing required columns: {missing}")

    x = frame[FEATURE_COLUMNS]
    y = frame[TARGET_COLUMN]
    return x, y


def print_missing_by_feature(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
) -> None:
    print("\nmissing values by feature")
    for column in FEATURE_COLUMNS:
        train_missing = int(train_x[column].isna().sum())
        test_missing = int(test_x[column].isna().sum())
        print(f"  {column}: train={train_missing:,}, test={test_missing:,}")


def evaluate_classification_metrics(
    y_true: pd.Series | np.ndarray,
    predicted_probability: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute classification metrics from predicted make probabilities."""
    y = np.asarray(y_true)
    predicted_label = predicted_probability >= threshold
    return {
        "auc": float(roc_auc_score(y, predicted_probability)),
        "log_loss": float(log_loss(y, predicted_probability)),
        "brier_score": float(brier_score_loss(y, predicted_probability)),
        "accuracy": float(accuracy_score(y, predicted_label)),
    }


def evaluate_baseline_metrics(
    y_true: pd.Series | np.ndarray,
    predicted_probability: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Alias for :func:`evaluate_classification_metrics` (backward compatible)."""
    return evaluate_classification_metrics(
        y_true,
        predicted_probability,
        threshold=threshold,
    )


def save_calibration_curve(
    y_true: pd.Series | np.ndarray,
    predicted_probability: np.ndarray,
    output_path: Path,
    *,
    model_label: str,
    title: str,
    n_bins: int = CALIBRATION_N_BINS,
) -> Path:
    """Plot predicted make probability vs. actual make rate by probability bin."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y_true)
    actual_make_rate, mean_predicted_prob = calibration_curve(
        y,
        predicted_probability,
        n_bins=n_bins,
        strategy="quantile",
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(
        mean_predicted_prob,
        actual_make_rate,
        marker="o",
        linewidth=2,
        label=model_label,
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Predicted make probability")
    ax.set_ylabel("Actual make rate")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_baseline_calibration_curve(
    y_true: pd.Series | np.ndarray,
    predicted_probability: np.ndarray,
    output_path: Path,
    *,
    n_bins: int = CALIBRATION_N_BINS,
) -> Path:
    """Baseline wrapper around :func:`save_calibration_curve`."""
    return save_calibration_curve(
        y_true,
        predicted_probability,
        output_path,
        model_label="Baseline logistic regression",
        title="Baseline calibration curve (test)",
        n_bins=n_bins,
    )


def save_probability_distribution(
    predicted_probability: np.ndarray,
    output_path: Path,
    *,
    title: str,
    bins: int = 50,
) -> Path:
    """Plot how spread out test predicted make probabilities are."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.hist(predicted_probability, bins=bins, edgecolor="black", alpha=0.75)
    ax.set_xlabel("Predicted make probability")
    ax.set_ylabel("Shot count")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_baseline_probability_distribution(
    predicted_probability: np.ndarray,
    output_path: Path,
    *,
    bins: int = 50,
) -> Path:
    """Baseline wrapper around :func:`save_probability_distribution`."""
    return save_probability_distribution(
        predicted_probability,
        output_path,
        title="Baseline predicted probability distribution (test)",
        bins=bins,
    )


def save_baseline_figures(
    y_true: pd.Series | np.ndarray,
    predicted_probability: np.ndarray,
    *,
    figures_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Write baseline calibration and probability distribution plots for the test set."""
    directory = figures_dir or DEFAULT_FIGURES_DIR
    calibration_path = save_baseline_calibration_curve(
        y_true,
        predicted_probability,
        directory / "baseline_calibration_curve.png",
    )
    distribution_path = save_baseline_probability_distribution(
        predicted_probability,
        directory / "baseline_probability_distribution.png",
    )
    return calibration_path, distribution_path
