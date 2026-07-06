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
    baseline_for_similar_shots,
    build_shot_edge_table,
    compare_prediction_to_baseline,
    compute_overview_stats,
    compute_shot_quality_summary,
    existing_model_artifact_paths,
    filter_shots,
    get_prediction_row,
    load_dashboard_splits,
    load_feature_importance,
    load_training_summary,
    prepare_prediction_features,
    sample_prediction_rows,
    summarize_by_distance_bucket,
    summarize_edge_backtest,
    top_feature_importance,
)
from courtvision.dashboard.prediction import (  # noqa: E402
    PREDICTION_UNAVAILABLE_MESSAGE,
    PredictionUnavailable,
    predict_prepared_shot,
)
from courtvision.dashboard.prediction import (  # noqa: E402
    create_model_service as _create_model_service,
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


@st.cache_resource
def _load_model_service():
    try:
        return _create_model_service()
    except PredictionUnavailable:
        return None


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


def _render_model_diagnostic_artifacts() -> None:
    st.markdown("#### Model diagnostic artifacts")

    artifacts = existing_model_artifact_paths()
    calibration_path = artifacts.get("calibration_curve")
    distribution_path = artifacts.get("probability_distribution")

    if calibration_path is None and distribution_path is None:
        st.warning(
            "No calibration or probability distribution artifacts found yet. Run LightGBM "
            "training with MLflow/artifact logging enabled to generate "
            "`reports/figures/lightgbm_calibration_curve.png` and "
            "`reports/figures/lightgbm_probability_distribution.png`."
        )
        return

    artifact_col1, artifact_col2 = st.columns(2)

    with artifact_col1:
        if calibration_path is not None:
            st.image(
                str(calibration_path),
                caption=(
                    "Calibration curve: checks whether predicted probabilities match "
                    "actual make rates."
                ),
            )
        else:
            st.caption("Calibration curve not available.")

    with artifact_col2:
        if distribution_path is not None:
            st.image(
                str(distribution_path),
                caption=(
                    "Probability distribution: shows how spread out the model's shot "
                    "probabilities are."
                ),
            )
        else:
            st.caption("Probability distribution not available.")


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
    _render_model_diagnostic_artifacts()


def _format_shot_result(made_flag: object) -> str:
    if pd.isna(made_flag):
        return "Unknown"
    return "Made" if bool(made_flag) else "Missed"


def _format_signed_points(value: float, *, precision: int = 2) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{precision}f}"


def _format_signed_percentage_points(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value * 100:.1f} percentage points"


def _comparison_interpretation(probability_edge: float) -> str:
    if probability_edge > 0:
        return (
            "For this selected shot, the model rates the attempt above the historical "
            "baseline for similar shots."
        )
    if probability_edge < 0:
        return (
            "For this selected shot, the model rates the attempt below the historical "
            "baseline for similar shots."
        )
    return (
        "For this selected shot, the model rates the attempt in line with the historical "
        "baseline for similar shots."
    )


def _render_prediction_comparison(
    test: pd.DataFrame,
    row: pd.Series,
    result,
) -> None:
    baseline = baseline_for_similar_shots(test, row)
    comparison = compare_prediction_to_baseline(result, row, baseline)

    st.markdown("#### Prediction comparison")
    st.caption("Model vs similar-shot baseline")

    compare_col1, compare_col2, compare_col3 = st.columns(3)
    with compare_col1:
        st.metric("Model make probability", f"{comparison.predicted_make_probability:.1%}")
        st.metric("Similar-shot baseline", f"{comparison.baseline_make_rate:.1%}")
        st.metric(
            "Probability difference",
            _format_signed_percentage_points(comparison.probability_edge_vs_baseline),
        )
    with compare_col2:
        st.metric("Model EV", f"{comparison.expected_shot_value:.2f}")
        st.metric("Baseline EV", f"{comparison.baseline_expected_value:.2f}")
        st.metric("EV difference", _format_signed_points(comparison.ev_edge_vs_baseline))
    with compare_col3:
        actual_points = comparison.actual_points
        actual_points_label = "—" if actual_points is None else f"{actual_points:.0f}"
        st.metric("Actual points", actual_points_label)
        st.metric("Similar shot count", f"{comparison.similar_shot_count:,}")

    st.info(_comparison_interpretation(comparison.probability_edge_vs_baseline))


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

    model_service = _load_model_service()
    if model_service is None:
        st.warning(PREDICTION_UNAVAILABLE_MESSAGE)

    if st.button("Predict make probability", disabled=model_service is None):
        try:
            result = predict_prepared_shot(prepared, service=model_service)
        except PredictionUnavailable:
            st.warning(PREDICTION_UNAVAILABLE_MESSAGE)
        except (KeyError, ValueError) as exc:
            st.error(str(exc))
        else:
            st.markdown("#### Prediction result")
            result_col1, result_col2, result_col3, result_col4 = st.columns(4)

            with result_col1:
                st.metric(
                    "Predicted make probability",
                    f"{result.predicted_make_probability:.1%}",
                )
            with result_col2:
                st.metric(
                    "Expected shot value",
                    f"{result.expected_shot_value:.2f}",
                )
            with result_col3:
                st.metric("Actual result", _format_shot_result(row[TARGET_COLUMN]))
            with result_col4:
                st.metric("Model", result.model_name)

            _render_prediction_comparison(test, row, result)


def _render_shot_edge_explorer(test: pd.DataFrame) -> None:
    st.subheader("Shot Edge Explorer")
    st.caption("Rank held-out test shots by model edge over the similar-shot baseline.")

    model_service = _load_model_service()
    if model_service is None:
        st.warning(PREDICTION_UNAVAILABLE_MESSAGE)
        return

    control_col1, control_col2, control_col3, control_col4 = st.columns(4)

    with control_col1:
        sample_size = int(
            st.selectbox("Sample size", options=["25", "50", "100"], index=1)
        )
    with control_col2:
        shot_value_label = st.selectbox("Shot value filter", options=["All", "2", "3"])
    with control_col3:
        sort_by = st.selectbox(
            "Sort by",
            options=[
                "EV edge",
                "Probability edge",
                "Expected shot value",
                "Predicted make probability",
            ],
        )
    with control_col4:
        direction = st.selectbox("Direction", options=["Highest first", "Lowest first"])

    sort_column_map = {
        "EV edge": "ev_edge_vs_baseline",
        "Probability edge": "probability_edge_vs_baseline",
        "Expected shot value": "expected_shot_value",
        "Predicted make probability": "predicted_make_probability",
    }
    ascending = direction == "Lowest first"

    if st.button("Score sample"):
        candidate_test = test
        if shot_value_label != "All":
            candidate_test = filter_shots(test, shot_value=int(shot_value_label))

        samples = sample_prediction_rows(candidate_test, n=sample_size)
        if samples.empty:
            st.warning("No test shots match the selected filters.")
            return

        predictions_by_row_id = {}
        errors: list[str] = []
        for row_id in samples[PREDICTION_ROW_ID_COLUMN]:
            try:
                prepared = prepare_prediction_features(get_prediction_row(test, row_id))
                predictions_by_row_id[int(row_id)] = predict_prepared_shot(
                    prepared,
                    service=model_service,
                )
            except (PredictionUnavailable, KeyError, ValueError) as exc:
                errors.append(f"Row {row_id}: {exc}")

        if errors:
            st.warning("Some shots could not be scored:\n" + "\n".join(errors))

        if not predictions_by_row_id:
            st.error("No shots were scored successfully.")
            return

        edge_table = build_shot_edge_table(test, predictions_by_row_id)
        edge_table = edge_table.sort_values(
            sort_column_map[sort_by],
            ascending=ascending,
        ).reset_index(drop=True)
        st.session_state["shot_edge_table"] = edge_table

    edge_table = st.session_state.get("shot_edge_table")
    if edge_table is None or edge_table.empty:
        st.info("Choose options and click **Score sample** to rank shots by model edge.")
        return

    if sort_by in sort_column_map:
        edge_table = edge_table.sort_values(
            sort_column_map[sort_by],
            ascending=ascending,
        ).reset_index(drop=True)

    summary_col1, summary_col2, summary_col3, summary_col4, summary_col5 = st.columns(5)

    with summary_col1:
        st.metric("Shots scored", f"{len(edge_table):,}")
    with summary_col2:
        st.metric(
            "Avg predicted make rate",
            f"{edge_table['predicted_make_probability'].mean():.1%}",
        )
    with summary_col3:
        st.metric("Avg model EV", f"{edge_table['expected_shot_value'].mean():.2f}")
    with summary_col4:
        st.metric(
            "Avg baseline EV",
            f"{edge_table['baseline_expected_value'].mean():.2f}",
        )
    with summary_col5:
        st.metric("Avg EV edge", f"{edge_table['ev_edge_vs_baseline'].mean():+.3f}")

    display_table = edge_table.copy()
    display_table["actual_made"] = display_table["actual_made"].map(
        lambda value: _format_shot_result(value) if value is not None else "Unknown"
    )
    display_table["predicted_make_probability"] = display_table[
        "predicted_make_probability"
    ].map(lambda value: f"{value:.1%}")
    display_table["baseline_make_rate"] = display_table["baseline_make_rate"].map(
        lambda value: f"{value:.1%}"
    )
    display_table["expected_shot_value"] = display_table["expected_shot_value"].map(
        lambda value: f"{value:.2f}"
    )
    display_table["baseline_expected_value"] = display_table["baseline_expected_value"].map(
        lambda value: f"{value:.2f}"
    )
    display_table["probability_edge_vs_baseline"] = display_table[
        "probability_edge_vs_baseline"
    ].map(_format_signed_percentage_points)
    display_table["ev_edge_vs_baseline"] = display_table["ev_edge_vs_baseline"].map(
        _format_signed_points
    )

    st.dataframe(display_table, use_container_width=True, hide_index=True)


def _weighted_backtest_average(summary: pd.DataFrame, column: str) -> float:
    active = summary.loc[summary["shot_count"] > 0]
    if active.empty:
        return 0.0
    total_shots = int(active["shot_count"].sum())
    return float((active[column] * active["shot_count"]).sum() / total_shots)


def _edge_backtest_interpretation(summary: pd.DataFrame) -> str:
    active = summary.loc[summary["shot_count"] > 0]
    if active.empty:
        return "Run a backtest sample to compare positive- and negative-edge buckets."

    positive = active.loc[active["bucket"].str.endswith("positive edge")]
    negative = active.loc[active["bucket"].str.endswith("negative edge")]
    if positive.empty or negative.empty:
        return (
            "This sample does not include both positive- and negative-edge buckets, "
            "so directional comparison is limited."
        )

    positive_make_rate = _weighted_backtest_average(positive, "actual_make_rate")
    negative_make_rate = _weighted_backtest_average(negative, "actual_make_rate")
    positive_points = _weighted_backtest_average(positive, "avg_actual_points")
    negative_points = _weighted_backtest_average(negative, "avg_actual_points")

    if positive_make_rate > negative_make_rate or positive_points > negative_points:
        return (
            "For this sample, positive-edge buckets show higher actual make rate or "
            "actual points than negative-edge buckets, so the model edge signal is "
            "behaving in the expected direction."
        )

    return (
        "For this sample, positive-edge buckets did not clearly outperform "
        "negative-edge buckets on actual make rate or actual points."
    )


def _render_edge_backtest(test: pd.DataFrame) -> None:
    st.subheader("Edge Backtest")
    st.caption(
        "Group scored held-out shots by model EV edge and compare model, baseline, "
        "and actual outcomes."
    )

    model_service = _load_model_service()
    if model_service is None:
        st.warning(PREDICTION_UNAVAILABLE_MESSAGE)
        return

    sample_size = int(st.selectbox("Sample size", options=["100", "250", "500"], index=0))

    if st.button("Run backtest"):
        samples = sample_prediction_rows(test, n=sample_size)
        if samples.empty:
            st.warning("No test shots available for backtest.")
            return

        predictions_by_row_id = {}
        errors: list[str] = []
        for row_id in samples[PREDICTION_ROW_ID_COLUMN]:
            try:
                prepared = prepare_prediction_features(get_prediction_row(test, row_id))
                predictions_by_row_id[int(row_id)] = predict_prepared_shot(
                    prepared,
                    service=model_service,
                )
            except (PredictionUnavailable, KeyError, ValueError) as exc:
                errors.append(f"Row {row_id}: {exc}")

        if errors:
            st.warning("Some shots could not be scored:\n" + "\n".join(errors))

        if not predictions_by_row_id:
            st.error("No shots were scored successfully.")
            return

        edge_table = build_shot_edge_table(test, predictions_by_row_id)
        backtest_summary = summarize_edge_backtest(edge_table)
        st.session_state["edge_backtest_table"] = edge_table
        st.session_state["edge_backtest_summary"] = backtest_summary

    edge_table = st.session_state.get("edge_backtest_table")
    backtest_summary = st.session_state.get("edge_backtest_summary")
    if edge_table is None or backtest_summary is None or edge_table.empty:
        st.info("Choose a sample size and click **Run backtest** to evaluate edge buckets.")
        return

    summary_col1, summary_col2, summary_col3, summary_col4, summary_col5 = st.columns(5)

    with summary_col1:
        st.metric("Shots scored", f"{len(edge_table):,}")
    with summary_col2:
        st.metric(
            "Avg predicted make rate",
            f"{edge_table['predicted_make_probability'].mean():.1%}",
        )
    with summary_col3:
        st.metric("Avg model EV", f"{edge_table['expected_shot_value'].mean():.2f}")
    with summary_col4:
        st.metric(
            "Avg baseline EV",
            f"{edge_table['baseline_expected_value'].mean():.2f}",
        )
    with summary_col5:
        st.metric("Avg EV edge", f"{edge_table['ev_edge_vs_baseline'].mean():+.3f}")

    display_summary = backtest_summary.copy()
    display_summary["avg_predicted_make_probability"] = display_summary[
        "avg_predicted_make_probability"
    ].map(lambda value: f"{value:.1%}")
    display_summary["actual_make_rate"] = display_summary["actual_make_rate"].map(
        lambda value: f"{value:.1%}"
    )
    for column in ("avg_model_ev", "avg_baseline_ev", "avg_actual_points"):
        display_summary[column] = display_summary[column].map(lambda value: f"{value:.2f}")
    signed_columns = (
        "avg_ev_edge",
        "model_ev_minus_actual_points",
        "baseline_ev_minus_actual_points",
    )
    for column in signed_columns:
        display_summary[column] = display_summary[column].map(_format_signed_points)

    st.markdown("#### Edge bucket backtest")
    st.dataframe(display_summary, use_container_width=True, hide_index=True)

    chart_data = backtest_summary.loc[backtest_summary["shot_count"] > 0].set_index("bucket")[
        ["avg_actual_points", "avg_model_ev", "avg_baseline_ev"]
    ]
    if not chart_data.empty:
        st.markdown("#### Actual points vs model EV vs baseline EV")
        st.bar_chart(chart_data)

    st.info(_edge_backtest_interpretation(backtest_summary))
    st.caption("This is a sampled diagnostic, not proof of future performance.")


def main() -> None:
    st.set_page_config(page_title="CourtVision Analytics", layout="wide")

    st.title("CourtVision Analytics Dashboard")
    st.caption(f"Season: {DEFAULT_SEASON}")

    st.markdown(
        "CourtVision predicts shot make probability and converts that probability "
        "into expected shot value."
    )

    train, test = _load_splits()

    overview_tab, explorer_tab, performance_tab, playground_tab, edge_tab, backtest_tab = st.tabs(
        [
            "Overview",
            "Shot Quality Explorer",
            "Model Performance",
            "Prediction Playground",
            "Shot Edge Explorer",
            "Edge Backtest",
        ]
    )

    with overview_tab:
        _render_overview(train, test)

    with explorer_tab:
        _render_shot_quality_explorer(test)

    with performance_tab:
        _render_model_performance()

    with playground_tab:
        _render_prediction_playground(test)

    with edge_tab:
        _render_shot_edge_explorer(test)

    with backtest_tab:
        _render_edge_backtest(test)


if __name__ == "__main__":
    main()
