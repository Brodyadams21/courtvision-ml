"""Smoke tests for FastAPI Docker packaging artifacts."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_api_exists() -> None:
    assert (PROJECT_ROOT / "Dockerfile.api").is_file()


def test_requirements_api_exists() -> None:
    assert (PROJECT_ROOT / "requirements-api.txt").is_file()


def test_dockerfile_api_runs_uvicorn_main_app() -> None:
    contents = (PROJECT_ROOT / "Dockerfile.api").read_text(encoding="utf-8")

    assert "uvicorn" in contents
    assert "courtvision.api.main:app" in contents


def test_dockerfile_api_exposes_port_8000() -> None:
    contents = (PROJECT_ROOT / "Dockerfile.api").read_text(encoding="utf-8")

    assert "EXPOSE 8000" in contents


def test_docker_compose_defines_api_service() -> None:
    contents = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "api:" in contents
    assert "Dockerfile.api" in contents


def test_docker_compose_maps_api_port_8000() -> None:
    contents = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert '"8000:8000"' in contents or "'8000:8000'" in contents or "8000:8000" in contents
