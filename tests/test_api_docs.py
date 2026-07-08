"""Smoke tests for CourtVision API documentation."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_doc(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_api_md_exists() -> None:
    assert (PROJECT_ROOT / "docs" / "api.md").is_file()


def test_api_deployment_md_exists() -> None:
    assert (PROJECT_ROOT / "docs" / "api_deployment.md").is_file()


def test_api_deployment_mentions_dockerfile_api() -> None:
    contents = _read_doc("docs/api_deployment.md")

    assert "Dockerfile.api" in contents


def test_api_deployment_mentions_mlflow_tracking_uri() -> None:
    contents = _read_doc("docs/api_deployment.md")

    assert "MLFLOW_TRACKING_URI" in contents


def test_api_deployment_mentions_remote_artifact_store() -> None:
    contents = _read_doc("docs/api_deployment.md")

    lowered = contents.lower()
    assert "artifact store" in lowered
    assert "s3" in lowered


def test_api_deployment_does_not_claim_production_deployed() -> None:
    contents = _read_doc("docs/api_deployment.md")
    lowered = contents.lower()

    assert "has not been deployed" in lowered or "not been deployed" in lowered
    assert "has been deployed to" not in lowered
    assert "is production ready" not in lowered
    assert "is production-ready" not in lowered


def test_api_md_links_to_deployment_guide() -> None:
    contents = _read_doc("docs/api.md")

    assert "api_deployment.md" in contents
