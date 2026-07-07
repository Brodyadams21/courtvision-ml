"""FastAPI application for CourtVision shot predictions."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request

from courtvision.api.logging import latency_ms, start_request_log
from courtvision.api.logging import logger as api_logger
from courtvision.api.model_service import ShotModelService
from courtvision.api.schemas import (
    BatchShotPredictionRequest,
    BatchShotPredictionResponse,
    ShotPredictionRequest,
    ShotPredictionResponse,
)
from courtvision.api.settings import load_api_settings


def get_model_service(request: Request) -> ShotModelService:
    """Return the model service attached to the application."""
    return request.app.state.model_service


def _log_prediction_error(request: Request, exc: BaseException) -> None:
    api_logger.warning(
        "prediction_error",
        extra={
            "request_id": request.state.request_id,
            "error_type": type(exc).__name__,
        },
    )


def create_app(
    *,
    model_service: ShotModelService | None = None,
    load_model_on_startup: bool = False,
    model_alias: str = "Candidate",
) -> FastAPI:
    """Build a FastAPI app; inject ``model_service`` for tests or custom wiring."""
    if model_service is None:
        service = ShotModelService()
        if load_model_on_startup:
            try:
                service.load_from_mlflow(alias=model_alias)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load MLflow model alias {model_alias!r} on startup: {exc}"
                ) from exc
    else:
        service = model_service

    app = FastAPI(title="CourtVision ML API", version="0.1.0")
    app.state.model_service = service

    @app.middleware("http")
    async def add_request_logging(request: Request, call_next):
        context = start_request_log()
        request.state.request_id = context.request_id
        try:
            response = await call_next(request)
        except Exception as exc:
            api_logger.exception(
                "request_failed",
                extra={
                    "request_id": context.request_id,
                    "path": request.url.path,
                    "method": request.method,
                    "status_code": 500,
                    "latency_ms": latency_ms(context),
                    "error_type": type(exc).__name__,
                },
            )
            raise
        response.headers["X-Request-ID"] = context.request_id
        api_logger.info(
            "request_completed",
            extra={
                "request_id": context.request_id,
                "path": request.url.path,
                "method": request.method,
                "status_code": response.status_code,
                "latency_ms": latency_ms(context),
            },
        )
        return response

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict/shot", response_model=ShotPredictionResponse)
    def predict_shot(
        body: ShotPredictionRequest,
        request: Request,
        service: Annotated[ShotModelService, Depends(get_model_service)],
    ) -> ShotPredictionResponse:
        try:
            prediction = service.predict_shot(body.features, body.shot_value)
        except RuntimeError as exc:
            _log_prediction_error(request, exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except KeyError as exc:
            _log_prediction_error(request, exc)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ValueError as exc:
            _log_prediction_error(request, exc)
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        api_logger.info(
            "single_prediction",
            extra={
                "request_id": request.state.request_id,
                "model_name": prediction.model_name,
                "batch_size": 1,
            },
        )
        return prediction

    @app.post("/predict/shots", response_model=BatchShotPredictionResponse)
    def predict_shots(
        body: BatchShotPredictionRequest,
        request: Request,
        service: Annotated[ShotModelService, Depends(get_model_service)],
    ) -> BatchShotPredictionResponse:
        shots = [(shot.features, shot.shot_value) for shot in body.shots]
        try:
            predictions = service.predict_shots(shots)
        except RuntimeError as exc:
            _log_prediction_error(request, exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except KeyError as exc:
            _log_prediction_error(request, exc)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ValueError as exc:
            _log_prediction_error(request, exc)
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        api_logger.info(
            "batch_prediction",
            extra={
                "request_id": request.state.request_id,
                "model_name": predictions[0].model_name if predictions else service.model_name,
                "batch_size": len(body.shots),
            },
        )
        return BatchShotPredictionResponse(predictions=predictions)

    return app


def create_app_from_env() -> FastAPI:
    settings = load_api_settings()
    return create_app(
        load_model_on_startup=settings.load_model_on_startup,
        model_alias=settings.model_alias,
    )


app = create_app_from_env()
