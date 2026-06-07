"""Single-feature leakage audit: train LightGBM on one column at a time and flag high AUC.

Includes ``score_margin`` (pre-shot margin when available) and ``score_margin_missing``
(whether margin was unavailable). High AUC on the missingness flag alone would suggest
missingness is label-correlated and worth investigating.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from courtvision.data.build_features import (
    DEFAULT_PROCESSED_FEATURES_DIR,
    processed_train_test_paths,
)
from courtvision.data.collect import DEFAULT_SEASON
from courtvision.models.common import (
    LEAKAGE_AUDIT_FEATURES,
    SCORE_MARGIN_FEATURE_COLUMNS,
    SUSPICIOUS_AUC_THRESHOLD,
    TARGET_COLUMN,
    evaluate_classification_metrics,
    load_train_test_parquet,
)
from courtvision.models.train_lgbm import train_and_predict_proba


def _require_score_margin_audit_columns(frame: pd.DataFrame) -> None:
    """Ensure parquet includes both score-margin audit columns."""
    missing = [col for col in SCORE_MARGIN_FEATURE_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(
            "Parquet is missing score-margin audit columns "
            f"{missing}. Rebuild/export features with fixed prior-PBP logic."
        )


def audit_single_feature(
    feature: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float | str | bool]:
    """Train on one feature and return test metrics."""
    train_x = x_train[[feature]]
    test_x = x_test[[feature]]
    _, test_proba = train_and_predict_proba(train_x, y_train, test_x)
    metrics = evaluate_classification_metrics(y_test, test_proba)
    auc = metrics["auc"]
    return {
        "feature": feature,
        "auc": auc,
        "log_loss": metrics["log_loss"],
        "brier_score": metrics["brier_score"],
        "accuracy": metrics["accuracy"],
        "train_missing_pct": 100.0 * train_x[feature].isna().mean(),
        "test_missing_pct": 100.0 * test_x[feature].isna().mean(),
        "suspicious": auc >= SUSPICIOUS_AUC_THRESHOLD,
    }


def print_audit_results(results: list[dict[str, float | str | bool]]) -> None:
    print(f"\nSingle-feature leakage audit (AUC >= {SUSPICIOUS_AUC_THRESHOLD:.2f} flagged)")
    print(f"{'feature':<40} {'auc':>8} {'log_loss':>10} {'accuracy':>10} {'flag':>6}")
    print("-" * 78)
    for row in results:
        flag = "YES" if row["suspicious"] else ""
        print(
            f"{row['feature']:<40} {row['auc']:8.4f} "
            f"{row['log_loss']:10.4f} {row['accuracy']:10.4f} {flag:>6}"
        )

    suspicious = [row for row in results if row["suspicious"]]
    if suspicious:
        print("\nSuspicious features:")
        for row in suspicious:
            print(
                f"  - {row['feature']}: auc={row['auc']:.4f}, "
                f"train_missing={row['train_missing_pct']:.1f}%, "
                f"test_missing={row['test_missing_pct']:.1f}%"
            )
    else:
        print("\nNo single feature exceeded the suspicious AUC threshold.")


def main(
    season: str = DEFAULT_SEASON,
    *,
    processed_dir: Path | None = None,
) -> None:
    paths = processed_train_test_paths(season, output_dir=processed_dir)
    print(f"Season: {season}")
    print(f"Train path: {paths['train']}")
    print(f"Test path:  {paths['test']}")
    print(f"Threshold:  AUC >= {SUSPICIOUS_AUC_THRESHOLD:.2f}")

    train, test = load_train_test_parquet(season, processed_dir=processed_dir)
    _require_score_margin_audit_columns(train)
    _require_score_margin_audit_columns(test)

    available = [col for col in LEAKAGE_AUDIT_FEATURES if col in train.columns]
    missing = [col for col in LEAKAGE_AUDIT_FEATURES if col not in train.columns]
    if missing:
        print(f"\nSkipping columns not in parquet: {missing}")

    y_train = train[TARGET_COLUMN]
    y_test = test[TARGET_COLUMN]
    x_train = train[available]
    x_test = test[available]

    print(f"\nAuditing {len(available)} features with LightGBM single-feature models")

    results: list[dict[str, float | str | bool]] = []
    for feature in available:
        results.append(
            audit_single_feature(feature, x_train, y_train, x_test, y_test),
        )

    results.sort(key=lambda row: float(row["auc"]), reverse=True)
    print_audit_results(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit shot-make features for target leakage via single-feature LightGBM runs.",
    )
    parser.add_argument("--season", default=DEFAULT_SEASON, help="Season label (e.g. 2024-25)")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_FEATURES_DIR,
        help=f"Directory with train/test Parquet (default: {DEFAULT_PROCESSED_FEATURES_DIR})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.season, processed_dir=args.processed_dir)
