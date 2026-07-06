"""CourtVision Analytics Dashboard — Streamlit entrypoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from courtvision.dashboard.data import (  # noqa: E402
    DEFAULT_FEATURE_IMPORTANCE_GAIN_PNG,
    PREDICTION_ROW_ID_COLUMN,
    compute_overview_stats,
    compute_shot_quality_summary,
    filter_shots,
    get_prediction_row,
    load_dashboard_splits,
    load_feature_importance,
    load_training_summary,
    prepare_prediction_features,
    sample_prediction_rows,
    summarize_by_distance_bucket,
    top_feature_importance,
)
from courtvision.data.collect import DEFAULT_SEASON  # noqa: E402
from courtvision.models.common import FEATURE_COLUMNS, TARGET_COLUMN  # noqa: E402


def _format_metric_value(value: float | None, *, precision: int = 4) -> str:
    if value is None:
        return "—"
    return f"{value:.{precision}f}"


def _format_percent_value(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.1%}"


@st.cache_data
def _load_cached_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_dashboard_splits(DEFAULT_SEASON)


def _load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        return _load_cached_splits()
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


def _render_feature_importance() -> None:
    st.markdown("#### Feature importance")

    try:
        importance = load_feature_importance()
    except ValueError as exc:
        st.error(str(exc))
        return

    if importance is None:
        st.warning(
            "No feature importance artifact found yet. Run LightGBM training with "
            "MLflow/artifact logging enabled to generate "
            "`reports/tables/lightgbm_feature_importance_gain.csv`."
        )
        return

    top_n_label = st.selectbox(
        "Top features to show",
        options=["10", "15", "20", "All"],
        index=1,
    )
    top_n = len(importance) if top_n_label == "All" else int(top_n_label)
    top_features = top_feature_importance(importance, n=top_n)

    chart_data = top_features.set_index("feature")[["importance"]]
    st.bar_chart(chart_data)

    with st.expander("Full feature importance table"):
        st.dataframe(importance, use_container_width=True, hide_index=True)

    if DEFAULT_FEATURE_IMPORTANCE_GAIN_PNG.is_file():
        st.image(str(DEFAULT_FEATURE_IMPORTANCE_GAIN_PNG), caption="LightGBM gain importance")


def _render_model_performance() -> None:
    st.subheader("Model Performance")
    st.caption("Held-out test metrics from the latest LightGBM training run.")

    summary = None
    try:
        summary = load_training_summary()
    except ValueError as exc:
        st.error(str(exc))

    if summary is None:
        st.warning(
            "No training summary found. Run model training to generate "
            "`model_artifacts/training_summary.json`."
        )
    else:
        metadata = pd.DataFrame(
            {
                "Field": ["Model", "Mode", "Environment", "Season", "Summary path"],
                "Value": [
                    summary.model,
                    summary.mode,
                    summary.environment,
                    summary.season,
                    str(summary.summary_path),
                ],
            }
        )
        st.dataframe(metadata, use_container_width=True, hide_index=True)

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        with metric_col1:
            st.metric("AUC", _format_metric_value(summary.auc))
        with metric_col2:
            st.metric("Log loss", _format_metric_value(summary.log_loss))
        with metric_col3:
            st.metric("Brier score", _format_metric_value(summary.brier_score))
        with metric_col4:
            st.metric("Accuracy", _format_percent_value(summary.accuracy))

        if summary.mode == "search" and any(
            value is not None
            for value in (
                summary.validation_auc,
                summary.validation_log_loss,
                summary.best_config_index,
            )
        ):
            st.markdown("#### Validation metrics")
            validation_col1, validation_col2, validation_col3 = st.columns(3)

            with validation_col1:
                st.metric("Validation AUC", _format_metric_value(summary.validation_auc))
            with validation_col2:
                st.metric(
                    "Validation log loss",
                    _format_metric_value(summary.validation_log_loss),
                )
            with validation_col3:
                best_config = (
                    str(summary.best_config_index)
                    if summary.best_config_index is not None
                    else "—"
                )
                st.metric("Best config index", best_config)

        with st.expander("Raw training summary JSON"):
            st.code(summary.summary_path.read_text(encoding="utf-8"), language="json")

    _render_feature_importance()


def _format_shot_result(made_flag: object) -> str:
    if pd.isna(made_flag):
        return "Unknown"
    return "Made" if bool(made_flag) else "Missed"


def _render_prediction_playground(test: pd.DataFrame) -> None:
    st.subheader("Prediction Playground")
    st.caption("Inspect real held-out test shots and prepare API-style model inputs.")

    samples = sample_prediction_rows(test)
    if samples.empty:
        st.warning("No test shots available for prediction preview.")
        return

    sample_lookup = samples.set_index(PREDICTION_ROW_ID_COLUMN)

    def _format_shot_option(row_id: int) -> str:
        sample_row = sample_lookup.loc[row_id]
        return (
            f"Row {row_id} | {int(sample_row['shot_value'])}pt | "
            f"{float(sample_row['shot_distance']):.1f} ft | "
            f"{_format_shot_result(sample_row[TARGET_COLUMN])}"
        )

    row_ids = samples[PREDICTION_ROW_ID_COLUMN].tolist()
    selected_row_id = st.selectbox(
        "Select a test shot",
        options=row_ids,
        format_func=_format_shot_option,
    )

    row = get_prediction_row(test, int(selected_row_id))
    prepared = prepare_prediction_features(row)

    st.markdown("#### Selected shot")
    summary_col1, summary_col2, summary_col3, summary_col4, summary_col5 = st.columns(5)

    with summary_col1:
        st.metric("Shot value", int(row["shot_value"]))
    with summary_col2:
        st.metric("Shot distance", f"{float(row['shot_distance']):.1f} ft")
    with summary_col3:
        st.metric("Period", int(row["period"]))
    with summary_col4:
        st.metric("Actual result", _format_shot_result(row[TARGET_COLUMN]))
    with summary_col5:
        st.metric("Score margin", f"{float(row['score_margin']):.1f}")

    location_col1, location_col2 = st.columns(2)
    with location_col1:
        st.metric("loc_x", f"{float(row['loc_x']):.1f}")
    with location_col2:
        st.metric("loc_y", f"{float(row['loc_y']):.1f}")

    feature_table = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "value": [row[column] for column in FEATURE_COLUMNS],
        }
    )
    st.markdown("#### Model features")
    st.dataframe(feature_table, use_container_width=True, hide_index=True)

    request_preview = {
        "features": prepared.features,
        "shot_value": prepared.shot_value,
    }
    st.markdown("#### Model input preview")
    st.code(json.dumps(request_preview, indent=2), language="json")

    st.info("Model prediction will be wired in the next task.")
    st.button("Predict make probability", disabled=True)


def main() -> None:
    st.set_page_config(page_title="CourtVision Analytics", layout="wide")

    st.title("CourtVision Analytics Dashboard")
    st.caption(f"Season: {DEFAULT_SEASON}")

    st.markdown(
        "CourtVision predicts shot make probability and converts that probability "
        "into expected shot value."
    )

    train, test = _load_splits()

    overview_tab, explorer_tab, performance_tab, playground_tab = st.tabs(
        ["Overview", "Shot Quality Explorer", "Model Performance", "Prediction Playground"]
    )

    with overview_tab:
        _render_overview(train, test)

    with explorer_tab:
        _render_shot_quality_explorer(test)

    with performance_tab:
        _render_model_performance()

    with playground_tab:
        _render_prediction_playground(test)


if __name__ == "__main__":
    main()
