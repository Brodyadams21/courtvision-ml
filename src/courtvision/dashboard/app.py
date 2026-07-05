"""CourtVision Analytics Dashboard — Streamlit entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import streamlit as st  # noqa: E402

from courtvision.dashboard.data import compute_overview_stats, load_dashboard_splits  # noqa: E402
from courtvision.data.collect import DEFAULT_SEASON  # noqa: E402


def main() -> None:
    st.set_page_config(page_title="CourtVision Analytics", layout="wide")

    st.title("CourtVision Analytics Dashboard")
    st.caption(f"Season: {DEFAULT_SEASON}")

    st.markdown(
        "CourtVision predicts shot make probability and converts that probability "
        "into expected shot value."
    )

    try:
        train, test = load_dashboard_splits(DEFAULT_SEASON)
        stats = compute_overview_stats(train, test)
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


if __name__ == "__main__":
    main()
