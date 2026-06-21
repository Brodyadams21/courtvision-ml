"""Storage path helpers for local and cloud workflows."""

from __future__ import annotations

S3_SCHEME = "s3://"


def is_s3_uri(value: str) -> bool:
    """Return True when ``value`` is an S3 URI."""
    return value.startswith(S3_SCHEME) and len(value) > len(S3_SCHEME)


def join_s3_uri(base_uri: str, *parts: str) -> str:
    """Join an S3 URI and key parts without duplicate slashes."""
    if not is_s3_uri(base_uri):
        raise ValueError(f"Expected S3 URI starting with {S3_SCHEME!r}: {base_uri!r}")

    base = base_uri.rstrip("/")
    cleaned_parts = [str(part).strip("/") for part in parts if str(part).strip("/")]
    if not cleaned_parts:
        return base
    return "/".join([base, *cleaned_parts])


def s3_uri(bucket: str, *parts: str) -> str:
    """Build an S3 URI from a bucket name and optional key parts."""
    cleaned_bucket = bucket.strip().removeprefix(S3_SCHEME).strip("/")
    if not cleaned_bucket:
        raise ValueError("S3 bucket name cannot be empty")
    return join_s3_uri(f"{S3_SCHEME}{cleaned_bucket}", *parts)
