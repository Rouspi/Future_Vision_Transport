# Module `fvt.data`

Rôle : chargement et préparation des données (Cityscapes).

Fichiers clés :
- `dataset_builder.py` : construction des datasets train/val (paths images/masques), intégration éventuelle de l’augmentation.
- `labels.py` : définition des classes, palettes et mapping pour la segmentation.

Chemins par défaut (dans `TrainingConfig`) :
- `train_images`, `train_masks`, `val_images`, `val_masks` pointent vers les dossiers Cityscapes.

Usage typique (extrait) :
```python
from fvt.training.pipeline import train_segmentation_model
from fvt.training.config import TrainingConfig
from fvt.config import load_settings

cfg = TrainingConfig(run_name="demo_run")
settings = load_settings()
train_segmentation_model(cfg, settings)
```

Notes :
- Les classes/poids sont alignés avec `labels.py`.
- L’augmentation est activable via `TrainingConfig.augment`.
