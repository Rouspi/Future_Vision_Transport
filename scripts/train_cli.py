#!/usr/bin/env python3
"""
CLI générique pour entraîner le modèle de segmentation.

Exemples rapides (adapter MLFLOW_TRACKING_URI) :
  PYTHONPATH=src MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \\
    python scripts/train_cli.py --preset mobilenet_midres --run-name mobilenet_ce_dice_midres

  PYTHONPATH=src MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \\
    python scripts/train_cli.py --preset vgg16_unet --run-name vgg16_unet_ce_dice

  PYTHONPATH=src MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \\
    python scripts/train_cli.py --preset unet_small_fullres --run-name unet_small_dice_fullres
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

from fvt.config import load_settings
from fvt.training.config import TrainingConfig
from fvt.training.pipeline import train_segmentation_model


def _parse_class_weights(value: Optional[str]) -> Optional[Tuple[float, ...]]:
    if value is None:
        return None
    return tuple(float(x) for x in value.split(","))


def _build_from_preset(name: Optional[str]) -> TrainingConfig:
    """Retourne une TrainingConfig préremplie selon un preset."""
    if name == "mobilenet_midres":
        return TrainingConfig(
            run_name="mobilenet_ce_dice_midres",
            model_type="mobilenetv2_deeplab_lite",
            input_height=256,
            input_width=512,
            batch_size=4,
            epochs=40,
            learning_rate=3e-4,
            loss_type="ce_dice",
            class_weights=(1.0, 1.0, 2.0, 1.0, 1.0, 2.5, 2.5),
            augment=True,
            fine_tune_from=80,
            mixed_precision=True,
            early_stopping_patience=8,
        )
    if name == "vgg16_unet":
        return TrainingConfig(
            run_name="vgg16_unet_ce_dice",
            model_type="vgg16_unet",
            input_height=256,
            input_width=512,
            batch_size=2,
            epochs=35,
            learning_rate=1e-4,
            loss_type="ce_dice",
            class_weights=(1.0, 1.0, 2.0, 1.0, 1.0, 2.5, 2.5),
            augment=True,
            fine_tune_from=15,
            mixed_precision=True,
            early_stopping_patience=6,
        )
    if name == "unet_small_fullres":
        return TrainingConfig(
            run_name="unet_small_dice_fullres",
            model_type="unet_small",
            input_height=512,
            input_width=1024,
            batch_size=2,
            epochs=30,
            learning_rate=2e-4,
            loss_type="ce_dice",
            class_weights=(1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0),
            augment=True,
            mixed_precision=True,
            early_stopping_patience=10,
        )
    return TrainingConfig()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraîne le modèle de segmentation via TrainingConfig.")
    parser.add_argument("--preset", choices=["mobilenet_midres", "vgg16_unet", "unet_small_fullres"],
                        help="Preset prêt à l'emploi ; peut être surchargé par les autres flags.")
    parser.add_argument("--run-name", help="Nom du run MLflow.")
    parser.add_argument("--model-type", choices=["mobilenetv2_deeplab_lite", "unet_small", "vgg16_unet"])
    parser.add_argument("--input-height", type=int)
    parser.add_argument("--input-width", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--validation-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--loss-type", choices=["ce_weighted", "dice", "ce_dice"])
    parser.add_argument("--class-weights", help="Poids de classes séparés par des virgules (ex: 1,1,2,1,1,2.5,2.5).")
    parser.add_argument("--augment", type=int, choices=[0, 1], help="1 pour activer, 0 pour désactiver.")
    parser.add_argument("--fine-tune-from", type=int, help="Index de couche jusqu'où geler le backbone.")
    parser.add_argument("--mixed-precision", type=int, choices=[0, 1], help="1 pour activer float16 mixte.")
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--train-images", type=Path, help="Chemin vers les images d'entraînement.")
    parser.add_argument("--train-masks", type=Path, help="Chemin vers les masques d'entraînement.")
    parser.add_argument("--val-images", type=Path, help="Chemin vers les images de validation.")
    parser.add_argument("--val-masks", type=Path, help="Chemin vers les masques de validation.")
    return parser.parse_args()


def apply_overrides(cfg: TrainingConfig, args: argparse.Namespace) -> TrainingConfig:
    # Simple override helper
    if args.run_name:
        cfg.run_name = args.run_name
    if args.model_type:
        cfg.model_type = args.model_type
    if args.input_height:
        cfg.input_height = args.input_height
    if args.input_width:
        cfg.input_width = args.input_width
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.epochs:
        cfg.epochs = args.epochs
    if args.steps_per_epoch is not None:
        cfg.steps_per_epoch = args.steps_per_epoch
    if args.validation_steps is not None:
        cfg.validation_steps = args.validation_steps
    if args.learning_rate:
        cfg.learning_rate = args.learning_rate
    if args.loss_type:
        cfg.loss_type = args.loss_type
    if args.class_weights:
        cfg.class_weights = _parse_class_weights(args.class_weights)
    if args.augment is not None:
        cfg.augment = bool(args.augment)
    if args.fine_tune_from is not None:
        cfg.fine_tune_from = args.fine_tune_from
    if args.mixed_precision is not None:
        cfg.mixed_precision = bool(args.mixed_precision)
    if args.early_stopping_patience is not None:
        cfg.early_stopping_patience = args.early_stopping_patience
    if args.train_images:
        cfg.train_images = args.train_images
    if args.train_masks:
        cfg.train_masks = args.train_masks
    if args.val_images:
        cfg.val_images = args.val_images
    if args.val_masks:
        cfg.val_masks = args.val_masks
    return cfg


def main() -> None:
    args = parse_args()
    cfg = _build_from_preset(args.preset)
    cfg = apply_overrides(cfg, args)

    settings = load_settings()
    print("Tracking URI:", settings.mlflow_tracking_uri)
    print("Run name:", cfg.run_name)
    print("Model type:", cfg.model_type)
    print("EarlyStopping patience (val_miou):", cfg.early_stopping_patience)
    train_segmentation_model(cfg, settings)


if __name__ == "__main__":
    main()
