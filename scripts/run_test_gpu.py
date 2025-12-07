#!/usr/bin/env python3
"""
Run de test GPU (courte durée) pour valider que le pipeline fonctionne sur le pod.
Usage:
  PYTHONPATH=src MLFLOW_TRACKING_URI=http://127.0.0.1:5000 python scripts/run_test_gpu.py
"""

import os
from pathlib import Path

from fvt.config import load_settings
from fvt.training.config import TrainingConfig
from fvt.training.pipeline import train_segmentation_model


def main() -> None:
    settings = load_settings()

    # Run court : résolution réduite, peu d'époques, pas de Registry.
    cfg = TrainingConfig(
        input_height=512,
        input_width=1024,
        batch_size=4,
        epochs=4,
        steps_per_epoch=300,
        validation_steps=80,
        model_type="vgg16_unet",
        loss_type="ce_dice",
        mixed_precision=True,
        run_name="test_gpu_short",
    )

    print("Tracking URI:", settings.mlflow_tracking_uri)
    print("Run name:", cfg.run_name)
    train_segmentation_model(cfg, settings)


if __name__ == "__main__":
    main()
