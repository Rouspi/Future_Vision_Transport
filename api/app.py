import hashlib
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from fvt.config import load_settings
from fvt.inference.predict import (
    InferenceConfig,
    load_model,
    mask_to_base64,
    predict_mask,
    preprocess_image,
)
from fvt.monitoring.mlflow_tracking import (
    configure_mlflow,
    load_model_by_stage,
    log_inference_event,
)

settings = load_settings()
app = FastAPI(title="Future Vision Segmentation API", version="0.1.0")

inference_cfg = InferenceConfig()
MODEL_SOURCE = os.getenv("MODEL_SOURCE", "local")
MODEL_PATH = Path(os.getenv("MODEL_PATH", settings.project_root / "artifacts" / "models" / "model.keras"))
MODEL_NAME = os.getenv("MODEL_NAME", "future_vision_segmentation")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

_MODEL_CACHE: Dict[str, Any] = {}


def _load_segmenter() -> Any:
    if _MODEL_CACHE.get("model") is not None:
        return _MODEL_CACHE["model"]

    if MODEL_SOURCE == "registry":
        configure_mlflow(settings, experiment=settings.mlflow_experiment)
        model = load_model_by_stage(MODEL_NAME, stage=MODEL_STAGE)
    else:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Local model not found at {MODEL_PATH}")
        model = load_model(MODEL_PATH)

    _MODEL_CACHE["model"] = model
    return model


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    start = time.perf_counter()
    content = await file.read()
    payload_hash = hashlib.sha256(content).hexdigest()[:16]

    try:
        with tempfile.NamedTemporaryFile(delete=True, suffix=Path(file.filename).suffix or ".png") as tmp:
            tmp.write(content)
            tmp.flush()
            image_tensor = preprocess_image(Path(tmp.name), inference_cfg.target_size())

        model = _load_segmenter()
        mask, _ = predict_mask(model, image_tensor, inference_cfg.num_classes)
        mask_b64 = mask_to_base64(mask)
    except Exception as exc:  # pylint: disable=broad-except
        latency_ms = (time.perf_counter() - start) * 1000
        log_inference_event(payload_hash, latency_ms, 500, model_name=MODEL_NAME, model_version=None)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency_ms = (time.perf_counter() - start) * 1000
    log_inference_event(payload_hash, latency_ms, 200, model_name=MODEL_NAME, model_version=None)

    return JSONResponse(
        {
            "mask_base64": mask_b64,
            "height": mask.shape[0],
            "width": mask.shape[1],
            "latency_ms": latency_ms,
            "payload_hash": payload_hash,
            "model_source": MODEL_SOURCE,
            "model_name": MODEL_NAME,
            "model_stage": MODEL_STAGE,
        }
    )
