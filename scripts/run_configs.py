#!/usr/bin/env python3
"""
Exécute une fois tous les fichiers .toml présents dans scripts/configs/inbox,
les déplace ensuite en processed/ ou failed/.

Usage :
  PYTHONPATH=src MLFLOW_TRACKING_URI=http://127.0.0.1:5000 python scripts/run_configs.py
Optionnel :
  --inbox        Chemin du dossier à lire (défaut: scripts/configs/inbox)
  --processed    Dossier où placer les configs réussies (défaut: scripts/configs/processed)
  --failed       Dossier où placer les configs en erreur (défaut: scripts/configs/failed)
"""

import argparse
import shutil
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
    base_dir = Path(__file__).resolve().parent / "configs"
    parser = argparse.ArgumentParser(description="Exécute une fois les configs TOML présentes dans inbox.")
    parser.add_argument("--inbox", type=Path, default=base_dir / "inbox", help="Répertoire des configs à exécuter.")
    parser.add_argument("--processed", type=Path, default=base_dir / "processed", help="Où déplacer en cas de succès.")
    parser.add_argument("--failed", type=Path, default=base_dir / "failed", help="Où déplacer en cas d'erreur.")
    args = parser.parse_args()

    inbox = args.inbox
    done_dir = args.processed
    failed_dir = args.failed
    for d in (inbox, done_dir, failed_dir):
        d.mkdir(parents=True, exist_ok=True)

    cfg_files = sorted(inbox.glob("*.toml"))
    if not cfg_files:
        print(f"Aucun fichier .toml trouvé dans {inbox}")
        return

    settings = load_settings()
    print("Tracking URI:", settings.mlflow_tracking_uri)
    print(f"Configs à exécuter : {len(cfg_files)} fichier(s)")

    for cfg_file in cfg_files:
        process_file(cfg_file, settings, done_dir, failed_dir)


if __name__ == "__main__":
    main()
