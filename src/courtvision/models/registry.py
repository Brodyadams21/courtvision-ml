"""MLflow model registry helpers for CourtVision shot-make models."""

from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient

REGISTERED_MODEL_NAME = "courtvision-shot-make-model"
CANDIDATE_ALIAS = "Candidate"
MODEL_ARTIFACT_PATH = "model"


def promote_model_version_to_candidate(
    version: int,
    *,
    registered_model_name: str = REGISTERED_MODEL_NAME,
    alias: str = CANDIDATE_ALIAS,
    client: MlflowClient | None = None,
) -> int:
    """Point the Candidate alias at a registered model version and tag it."""
    registry_client = client or MlflowClient()
    registry_client.set_registered_model_alias(
        name=registered_model_name,
        alias=alias,
        version=version,
    )
    registry_client.set_registered_model_tag(
        name=registered_model_name,
        key="candidate",
        value="true",
    )
    registry_client.set_model_version_tag(
        name=registered_model_name,
        version=str(version),
        key="role",
        value=alias,
    )
    return version


def register_run_model_as_candidate(
    run_id: str,
    *,
    artifact_path: str = MODEL_ARTIFACT_PATH,
    registered_model_name: str = REGISTERED_MODEL_NAME,
    alias: str = CANDIDATE_ALIAS,
    client: MlflowClient | None = None,
) -> int:
    """Register a logged run model and promote it to the Candidate alias."""
    model_uri = f"runs:/{run_id}/{artifact_path}"
    model_version = mlflow.register_model(model_uri, registered_model_name)
    version = int(model_version.version)
    return promote_model_version_to_candidate(
        version,
        registered_model_name=registered_model_name,
        alias=alias,
        client=client,
    )
