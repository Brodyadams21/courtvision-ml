"""Tests for MLflow registry helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

from courtvision.models.registry import (
    CANDIDATE_ALIAS,
    REGISTERED_MODEL_NAME,
    promote_model_version_to_candidate,
)


def test_promote_model_version_to_candidate_sets_alias_and_tags() -> None:
    client = MagicMock()

    version = promote_model_version_to_candidate(
        3,
        client=client,
    )

    assert version == 3
    client.set_registered_model_alias.assert_called_once_with(
        name=REGISTERED_MODEL_NAME,
        alias=CANDIDATE_ALIAS,
        version=3,
    )
    client.set_registered_model_tag.assert_called_once_with(
        name=REGISTERED_MODEL_NAME,
        key="candidate",
        value="true",
    )
    client.set_model_version_tag.assert_called_once_with(
        name=REGISTERED_MODEL_NAME,
        version="3",
        key="role",
        value=CANDIDATE_ALIAS,
    )
