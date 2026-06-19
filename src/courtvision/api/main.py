"""FastAPI application for CourtVision shot predictions."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request

from courtvision.api.model_service import ShotModelService
from courtvision.api.schemas import ShotPredictionRequest, ShotPredictionResponse


def get_model_service(request: Request) -> ShotModelService:
    """Return the model service attached to the application."""
    return request.app.state.model_service


def create_app(*, model_service: ShotModelService | None = None) -> FastAPI:
    """Build a FastAPI app; inject ``model_service`` for tests or custom wiring."""
    app = FastAPI(title="CourtVision ML API", version="0.1.0")
    app.state.model_service = model_service or ShotModelService()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict/shot", response_model=ShotPredictionResponse)
    def predict_shot(
        body: ShotPredictionRequest,
        service: Annotated[ShotModelService, Depends(get_model_service)],
    ) -> ShotPredictionResponse:
        try:
            return service.predict_shot(body.features, body.shot_value)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


app = create_app()
