# Librairie `fvt`

Contenu principal :
- `data/` : construction des datasets (Cityscapes), labels, loaders.
- `training/` : config (`TrainingConfig`), pipeline d’entraînement, callbacks (TopK checkpoints), losses/métriques.
- `utils/` : métriques (Dice, mIoU), helpers MLflow, outils divers.
- `models/` : architectures (`mobilenetv2_deeplab_lite`, `unet_small`, `vgg16_unet`).
- `inference/` : prédiction sur images.

Exemple d’usage minimal :
```python
from fvt.training.pipeline import train_segmentation_model
from fvt.training.config import TrainingConfig
from fvt.config import load_settings

cfg = TrainingConfig(run_name="demo_run")
settings = load_settings()
train_segmentation_model(cfg, settings)
```

Entrées/sorties :
- Chemins par défaut Cityscapes (train/val) définis dans `TrainingConfig`.
- Artefacts : modèles/checkpoints en `artifacts/` suffixés par `run_name`, loggés aussi dans MLflow si configuré.
- Paramètre `top_k_checkpoints` : None garde tous les checkpoints, sinon conserve les k meilleurs sur `val_miou`.
