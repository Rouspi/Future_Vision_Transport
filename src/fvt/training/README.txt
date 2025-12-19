# Module `fvt.training`

Rôle : configuration et pipeline d’entraînement.

Fichiers clés :
- `config.py` : dataclass `TrainingConfig` (dimensions, modèle, loss, chemins datasets, checkpoints, etc.).
- `pipeline.py` : fonction `train_segmentation_model` (callbacks, MLflow, checkpoints top-K, logging modèle).
- `__init__.py` : exports.

Paramètres principaux (TrainingConfig) :
- `run_name`, `experiment_name`.
- `input_height`, `input_width`, `batch_size`, `epochs`, `learning_rate`, `augment`, `mixed_precision`.
- `model_type` : `mobilenetv2_deeplab_lite`, `unet_small`, `vgg16_unet`.
- `loss_type` : `ce_weighted`, `dice`, `ce_dice`.
- `class_weights` : poids par classe (7 classes).
- `top_k_checkpoints` : None = tous les ckpts, sinon conserve les K meilleurs `val_miou`.
- Chemins Cityscapes par défaut ; surcharge possibles (`train_images`, `train_masks`, etc.).

Usage typique :
```python
from fvt.training.pipeline import train_segmentation_model
from fvt.training.config import TrainingConfig
from fvt.config import load_settings

cfg = TrainingConfig(run_name="demo_run")
settings = load_settings()
train_segmentation_model(cfg, settings)
```

Notes :
- Les artefacts sont écrits sous `artifacts/models/<run_name>/` et `artifacts/checkpoints/<run_name>/`, et loggés dans MLflow si `MLFLOW_TRACKING_URI` est défini.
- Callbacks : EarlyStopping, ReduceLROnPlateau, TopK checkpoints (k configurable). Le meilleur checkpoint est rechargé avant le log du modèle.
