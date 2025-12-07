from __future__ import annotations

import time
from typing import Any, Dict, Optional

import mlflow
from mlflow.tracking import MlflowClient

from fvt.config import Settings


def configure_mlflow(settings: Settings, experiment: Optional[str] = None) -> MlflowClient:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment or settings.mlflow_experiment)
    return MlflowClient(tracking_uri=settings.mlflow_tracking_uri)


def load_model_by_stage(model_name: str, stage: str = "Production") -> Any:
    model_uri = f"models:/{model_name}/{stage}"
    return mlflow.pyfunc.load_model(model_uri=model_uri)


def log_inference_event(
    payload_hash: str,
    latency_ms: float,
    status_code: int,
    model_name: str,
    model_version: Optional[str] = None,
    experiment: str = "inference-logs",
    extra_tags: Optional[Dict[str, str]] = None,
) -> None:
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name="api_inference", nested=True):
        mlflow.log_metrics({"latency_ms": latency_ms})
        mlflow.set_tags(
            {
                "payload_hash": payload_hash,
                "status_code": status_code,
                "model_name": model_name,
                "model_version": model_version or "unknown",
                "event_ts": int(time.time()),
                **(extra_tags or {}),
            }
        )
