"""Tests for leakage audit feature coverage."""

from __future__ import annotations

import pandas as pd
import pytest

from courtvision.models.audit_feature_leakage import _require_score_margin_audit_columns
from courtvision.models.common import (
    FEATURE_COLUMNS,
    LEAKAGE_AUDIT_FEATURES,
    SCORE_MARGIN_FEATURE_COLUMNS,
)


def test_leakage_audit_includes_score_margin_columns() -> None:
    for column in SCORE_MARGIN_FEATURE_COLUMNS:
        assert column in LEAKAGE_AUDIT_FEATURES


def test_score_margin_columns_are_in_feature_columns() -> None:
    for column in SCORE_MARGIN_FEATURE_COLUMNS:
        assert column in FEATURE_COLUMNS


def test_require_score_margin_audit_columns_passes_when_present() -> None:
    frame = pd.DataFrame(
        {
            "score_margin": [0, 1],
            "score_margin_missing": [False, True],
        }
    )
    _require_score_margin_audit_columns(frame)


def test_require_score_margin_audit_columns_raises_when_missing() -> None:
    frame = pd.DataFrame({"score_margin": [0, 1]})
    with pytest.raises(ValueError, match="score_margin_missing"):
        _require_score_margin_audit_columns(frame)
