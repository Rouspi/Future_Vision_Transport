#!/usr/bin/env python3
"""
Log d'un modèle Keras local dans un run MLflow existant.

Usage (exemple) :
  PYTHONPATH=src MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \\
    python scripts/log_model_to_run.py \\
      --run-id 123abc... \\
      --model-path artifacts/models/unet_small_ce_dice_extra/model.keras \\
      --loss-type ce_dice

Notes :
- On recharge le modèle avec les objets custom (DiceMetric + loss) puis on le loggue
  dans le run ciblé (pas de modification des métriques/durée).
- L'expérience n'est pas modifiée ; on se contente d'ajouter l'artefact `model/`.
"""

import argparse
from pathlib import Path

import mlflow
import numpy as np
import tensorflow as tf
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

from fvt.utils.losses import DiceMetric, build_loss


def log_model_to_run(run_id: str, model_path: Path, loss_type: str) -> None:
    loss_fn = build_loss(loss_type)
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"DiceMetric": DiceMetric, loss_fn.__name__: loss_fn},
    )

    input_shape = model.input_shape[1:]
    output_shape = model.output_shape[1:]
    signature = ModelSignature(
        inputs=Schema([TensorSpec(np.dtype("float32"), (-1,) + tuple(input_shape))]),
        outputs=Schema([TensorSpec(np.dtype("float32"), (-1,) + tuple(output_shape))]),
    )

    with mlflow.start_run(run_id=run_id):
        mlflow.keras.log_model(
            model,
            artifact_path="model",
            signature=signature,
            pip_requirements=[
                f"tensorflow=={tf.__version__}",
                f"keras=={tf.keras.__version__}",
                "cloudpickle",
            ],
        )
    print(f"Modèle logué dans le run {run_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log d'un modèle local dans un run MLflow existant.")
    parser.add_argument("--run-id", required=True, help="ID du run MLflow cible.")
    parser.add_argument(
        "--model-path",
        required=True,
        type=Path,
        help="Chemin du fichier model.keras à logguer.",
    )
    parser.add_argument(
        "--loss-type",
        default="ce_dice",
        choices=["ce_dice", "ce_weighted", "dice"],
        help="Loss utilisée pour construire les custom_objects lors du chargement.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {args.model_path}")
    log_model_to_run(run_id=args.run_id, model_path=args.model_path, loss_type=args.loss_type)


if __name__ == "__main__":
    main()
