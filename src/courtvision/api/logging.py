"""Structured logging helpers for the CourtVision inference API."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass

logger = logging.getLogger("courtvision.api")


@dataclass(frozen=True)
class RequestLogContext:
    request_id: str
    started_at: float


def start_request_log() -> RequestLogContext:
    return RequestLogContext(
        request_id=str(uuid.uuid4()),
        started_at=time.perf_counter(),
    )


def latency_ms(context: RequestLogContext) -> float:
    return (time.perf_counter() - context.started_at) * 1000.0
