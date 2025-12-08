#!/usr/bin/env python3
"""
Watcher de configs TOML : lit les fichiers qui arrivent dans un répertoire,
entraîne, puis déplace en processed/ ou failed/.

Usage (exemple) :
  PYTHONPATH=src MLFLOW_TRACKING_URI=http://127.0.0.1:5000 python scripts/watch_configs.py \\
    --inbox configs/inbox --processed configs/processed --failed configs/failed --poll-interval 10

Format TOML attendu (exemple) à déposer dans inbox/ :
  run_name = "mobilenet_ce_dice_midres"
  model_type = "mobilenetv2_deeplab_lite"
  input_height = 256
  input_width = 512
  batch_size = 4
  epochs = 40
  learning_rate = 0.0003
  loss_type = "ce_dice"
  class_weights = [1.0, 1.0, 2.0, 1.0, 1.0, 2.5, 2.5]
  augment = true
  fine_tune_from = 80
  mixed_precision = true
  early_stopping_patience = 8
  # chemins optionnels si tu veux override
  # train_images = "data/..."
  # train_masks  = "data/..."
  # val_images   = "data/..."
  # val_masks    = "data/..."
"""

import argparse
import shutil
import time
from pathlib import Path
from typing import Any, Dict

try:
    import tomllib  # Py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from fvt.config import load_settings
from fvt.training.config import TrainingConfig
from fvt.training.pipeline import train_segmentation_model


def load_cfg(path: Path) -> TrainingConfig:
    data: Dict[str, Any] = tomllib.loads(path.read_text())
    return TrainingConfig(**data)


def process_file(cfg_file: Path, settings, done_dir: Path, failed_dir: Path) -> None:
    print(f"==> Running config {cfg_file.name}")
    try:
        cfg = load_cfg(cfg_file)
        train_segmentation_model(cfg, settings)
        shutil.move(str(cfg_file), done_dir / cfg_file.name)
        print(f"Moved to {done_dir / cfg_file.name}")
    except Exception as e:  # pragma: no cover - runtime guard
        print(f"Error on {cfg_file.name}: {e}")
        shutil.move(str(cfg_file), failed_dir / cfg_file.name)
        print(f"Moved to {failed_dir / cfg_file.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a folder for TOML configs and train on arrival.")
    base_dir = Path(__file__).resolve().parent / "configs"
    parser.add_argument("--inbox", type=Path, default=base_dir / "inbox", help="Répertoire à surveiller (défaut: scripts/configs/inbox).")
    parser.add_argument("--processed", type=Path, default=base_dir / "processed", help="Où déplacer en cas de succès (défaut: scripts/configs/processed).")
    parser.add_argument("--failed", type=Path, default=base_dir / "failed", help="Où déplacer en cas d'erreur (défaut: scripts/configs/failed).")
    parser.add_argument("--poll-interval", type=int, default=10, help="Intervalle de polling (secondes).")
    args = parser.parse_args()

    inbox = args.inbox
    done_dir = args.processed
    failed_dir = args.failed
    for d in (inbox, done_dir, failed_dir):
        d.mkdir(parents=True, exist_ok=True)

    settings = load_settings()
    print("Tracking URI:", settings.mlflow_tracking_uri)
    print("Watching:", inbox.resolve())

    while True:
        for cfg_file in sorted(inbox.glob("*.toml")):
            process_file(cfg_file, settings, done_dir, failed_dir)
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
