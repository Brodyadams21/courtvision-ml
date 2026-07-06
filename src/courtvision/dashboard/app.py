"""CourtVision Analytics Dashboard — Streamlit entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from courtvision.dashboard.data import (  # noqa: E402
    compute_overview_stats,
    compute_shot_quality_summary,
    filter_shots,
    load_dashboard_splits,
    load_training_summary,
    summarize_by_distance_bucket,
)
from courtvision.data.collect import DEFAULT_SEASON  # noqa: E402


def _load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        return load_dashboard_splits(DEFAULT_SEASON)
    except FileNotFoundError as exc:
        st.error(
            "Could not load train/test feature files. "
            "Run the local feature pipeline first to generate processed Parquet splits."
        )
        st.code(str(exc))
        st.stop()
    except KeyError as exc:
        st.error(f"Feature data is missing required columns: {exc}")
        st.stop()


def _render_overview(train: pd.DataFrame, test: pd.DataFrame) -> None:
    stats = compute_overview_stats(train, test)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Train shots", f"{stats.train_shots:,}")
        st.metric("Train make rate", f"{stats.train_make_rate:.1%}")

    with col2:
        st.metric("Test shots", f"{stats.test_shots:,}")
        st.metric("Test make rate", f"{stats.test_make_rate:.1%}")

    with col3:
        st.metric("Total shots", f"{stats.total_shots:,}")
        st.metric("Feature count", stats.feature_count)


def _render_shot_quality_explorer(test: pd.DataFrame) -> None:
    st.subheader("Shot Quality Explorer")
    st.caption("Explore held-out test shots by shot type, distance, and period.")

    min_distance = float(test["shot_distance"].min())
    max_distance = float(test["shot_distance"].max())
    available_periods = sorted(int(period) for period in test["period"].dropna().unique())

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        shot_value_label = st.selectbox(
            "Shot value",
            options=["All", "2", "3"],
            index=0,
        )
    with filter_col2:
        selected_periods = st.multiselect(
            "Periods",
            options=available_periods,
            default=available_periods,
        )
    with filter_col3:
        distance_range = st.slider(
            "Shot distance (ft)",
            min_value=min_distance,
            max_value=max_distance,
            value=(min_distance, max_distance),
        )

    shot_value = None if shot_value_label == "All" else int(shot_value_label)
    periods = selected_periods if selected_periods else None

    filtered = filter_shots(
        test,
        shot_value=shot_value,
        periods=periods,
        min_distance=distance_range[0],
        max_distance=distance_range[1],
    )
    summary = compute_shot_quality_summary(filtered)
    bucket_summary = summarize_by_distance_bucket(filtered)

    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

    with metric_col1:
        st.metric("Filtered shots", f"{summary.shot_count:,}")
    with metric_col2:
        st.metric("Make rate", f"{summary.make_rate:.1%}")
    with metric_col3:
        st.metric("Avg shot value", f"{summary.avg_shot_value:.2f}")
    with metric_col4:
        st.metric("Avg distance", f"{summary.avg_shot_distance:.1f} ft")
    with metric_col5:
        st.metric("Baseline expected pts", f"{summary.avg_expected_points_baseline:.3f}")

    st.markdown("#### By distance bucket")
    display_table = bucket_summary.copy()
    display_table["make_rate"] = display_table["make_rate"].map(lambda value: f"{value:.1%}")
    display_table["avg_shot_value"] = display_table["avg_shot_value"].map(
        lambda value: f"{value:.2f}"
    )
    display_table["avg_expected_points_baseline"] = display_table[
        "avg_expected_points_baseline"
    ].map(lambda value: f"{value:.3f}")
    st.dataframe(display_table, use_container_width=True, hide_index=True)

    chart_data = bucket_summary.set_index("distance_bucket")[
        ["make_rate", "avg_expected_points_baseline"]
    ]
    st.markdown("#### Make rate and baseline expected points by distance")
    st.bar_chart(chart_data)


def _render_model_performance() -> None:
    st.subheader("Model Performance")
    st.caption("Held-out test metrics from the latest LightGBM training run.")

    try:
        summary = load_training_summary()
    except ValueError as exc:
        st.error(str(exc))
        return

    if summary is None:
        st.warning(
            "No training summary found. Run model training to generate "
            "`model_artifacts/training_summary.json`."
        )
        return

    meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)

    with meta_col1:
        st.metric("Model", summary.model)
    with meta_col2:
        st.metric("Mode", summary.mode)
    with meta_col3:
        st.metric("Environment", summary.environment)
    with meta_col4:
        st.metric("Season", summary.season)

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric("AUC", f"{summary.auc:.4f}")
    with metric_col2:
        st.metric("Log loss", f"{summary.log_loss:.4f}")
    with metric_col3:
        st.metric("Brier score", f"{summary.brier_score:.4f}")
    with metric_col4:
        st.metric("Accuracy", f"{summary.accuracy:.1%}")

    with st.expander("Raw training summary JSON"):
        st.code(summary.summary_path.read_text(encoding="utf-8"), language="json")


def main() -> None:
    st.set_page_config(page_title="CourtVision Analytics", layout="wide")

    st.title("CourtVision Analytics Dashboard")
    st.caption(f"Season: {DEFAULT_SEASON}")

    st.markdown(
        "CourtVision predicts shot make probability and converts that probability "
        "into expected shot value."
    )

    train, test = _load_splits()

    overview_tab, explorer_tab, performance_tab = st.tabs(
        ["Overview", "Shot Quality Explorer", "Model Performance"]
    )

    with overview_tab:
        _render_overview(train, test)

    with explorer_tab:
        _render_shot_quality_explorer(test)

    with performance_tab:
        _render_model_performance()


if __name__ == "__main__":
    main()
