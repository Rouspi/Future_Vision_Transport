#!/usr/bin/env python3
import argparse
from pathlib import Path
from PIL import Image

from fvt.training.config import TrainingConfig
from fvt.training.pipeline import build_model
from fvt.inference.predict import (
    load_model as load_savedmodel,
    preprocess_image,
    predict_mask,
)
from fvt.data.labels import colorize

def load_segmenter(model_path: Path, cfg: TrainingConfig):
    if model_path.is_dir():
        # SavedModel complet
        return load_savedmodel(model_path)
    # Sinon, checkpoint de poids
    model = build_model(cfg)
    model.load_weights(model_path)
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Chemin image d'entrée (PNG/JPG)")
    ap.add_argument("--model", default="artifacts/models/saved_model",
                    help="SavedModel (dossier) ou checkpoint .weights.h5")
    ap.add_argument("--out", default="artifacts/pred_mask.png",
                    help="Chemin de sortie pour le mask colorisé")
    args = ap.parse_args()

    cfg = TrainingConfig()
    model = load_segmenter(Path(args.model), cfg)
    image_tensor = preprocess_image(Path(args.image), (cfg.input_height, cfg.input_width))
    mask_idx, _ = predict_mask(model, image_tensor, num_classes=cfg.num_classes)
    Image.fromarray(colorize(mask_idx)).save(args.out)
    print(f"Mask sauvegardé dans {args.out}")

if __name__ == "__main__":
    main()

