"""Tests for storage path helpers."""

from __future__ import annotations

import pytest

from courtvision.utils.storage import is_s3_uri, join_s3_uri, s3_uri


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("s3://bucket/key", True),
        ("s3://bucket", True),
        ("data/processed/features", False),
        ("", False),
        ("https://bucket/key", False),
    ],
)
def test_is_s3_uri(value: str, expected: bool) -> None:
    assert is_s3_uri(value) is expected


def test_join_s3_uri_joins_parts_cleanly() -> None:
    result = join_s3_uri("s3://courtvision-bucket/", "/processed/", "features/")

    assert result == "s3://courtvision-bucket/processed/features"


def test_join_s3_uri_returns_base_when_no_parts() -> None:
    assert join_s3_uri("s3://courtvision-bucket/") == "s3://courtvision-bucket"


def test_join_s3_uri_rejects_non_s3_base() -> None:
    with pytest.raises(ValueError, match="Expected S3 URI"):
        join_s3_uri("data/processed", "features")


def test_s3_uri_builds_from_bucket_name() -> None:
    result = s3_uri("courtvision-bucket", "processed", "features")

    assert result == "s3://courtvision-bucket/processed/features"


def test_s3_uri_accepts_bucket_with_scheme() -> None:
    result = s3_uri("s3://courtvision-bucket", "processed/features")

    assert result == "s3://courtvision-bucket/processed/features"


def test_s3_uri_rejects_empty_bucket() -> None:
    with pytest.raises(ValueError, match="bucket name cannot be empty"):
        s3_uri("")
