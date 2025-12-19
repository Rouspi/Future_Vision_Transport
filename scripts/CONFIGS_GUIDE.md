# Guide des configs TOML (scripts/run_configs.py)

## Principe
- Placer vos fichiers `.toml` dans `scripts/configs/inbox/`.
- Lancer `scripts/run_configs.py` : chaque fichier est exécuté, puis déplacé en `processed/` ou `failed/`.
- Le fichier `.toml` mappe directement sur les champs de `TrainingConfig` (cf. `src/fvt/training/config.py`).

## Exemple minimal
```
run_name = "mobilenetv2_cityscapes"
experiment_name = "cityscapes"

input_height = 512
input_width  = 1024
batch_size   = 2
epochs       = 20
learning_rate = 1e-4
augment = true
mixed_precision = false

model_type = "mobilenetv2_deeplab_lite"  # ou unet_small, vgg16_unet
loss_type  = "ce_weighted"               # ou dice, ce_dice
class_weights = [1.0, 1.0, 1.5, 1.0, 1.0, 2.0, 2.0]

# Optionnel : nombre de checkpoints conservés (None = tous)
# top_k_checkpoints = 3
```

## Principaux paramètres
- `run_name` : suffixe pour les artefacts (modèles/ckpts), nom du run MLflow.
- `experiment_name` : expérience MLflow ciblée (défaut dans TrainingConfig : `future_vision_segmentation`).
- `input_height`, `input_width` : dimensions d’entrée (le modèle attend ce format exact).
- `batch_size`, `epochs`, `learning_rate`, `augment`, `mixed_precision`.
- `model_type` : `mobilenetv2_deeplab_lite`, `unet_small`, `vgg16_unet`.
- `loss_type` : `ce_weighted`, `dice`, `ce_dice`.
- `class_weights` : tuple/list de 7 poids (ordre des classes défini dans labels.py).
- `top_k_checkpoints` : None pour garder tous les checkpoints, sinon conserve les K meilleurs `val_miou`.
- Chemins par défaut Cityscapes (train/val) sont dans `TrainingConfig` ; surcharger si besoin :
  - `train_images`, `train_masks`, `val_images`, `val_masks`.

## Lancer les configs
```bash
PYTHONPATH=src MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
python scripts/run_configs.py --inbox scripts/configs/inbox
```

## Artefacts
- Modèles : `artifacts/models/<run_name>/model.keras` (meilleur checkpoint rechargé).
- Checkpoints : `artifacts/checkpoints/<run_name>/` (tous ou top-K).
- MLflow : artefacts loggés vers le `artifact_uri` du run (assurez-vous que MLFLOW_TRACKING_URI pointe vers le bon serveur).

## Remarques
- Pour éviter l’écrasement, garder des `run_name` uniques.
- Vérifier que `MLFLOW_TRACKING_URI` est joignable avant de lancer (ex : tunnel SSH si serveur distant).
- Sur Mac Intel, pip ne distribue pas TF > 2.16.2 ; adapter l’environnement si besoin pour charger les modèles Keras 3.
