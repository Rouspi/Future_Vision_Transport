# Guide rapide des scripts

## run_configs.py
Exécute en batch les configs TOML placées dans `scripts/configs/inbox/`, puis déplace en `processed/` ou `failed/`.
```bash
PYTHONPATH=src MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
python scripts/run_configs.py --inbox scripts/configs/inbox
```
Voir aussi `CONFIGS_GUIDE.md` pour le format TOML.

## train_cli.py
Entrée unique pour lancer un entraînement à partir d’un fichier TOML ou de paramètres en CLI.
Exemple :
```bash
PYTHONPATH=src MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
python scripts/train_cli.py --config scripts/configs/inbox/unet_small_ce_dice_extra.toml
```

## watch_configs.py
Boucle de surveillance : lit en continu un répertoire (défaut `scripts/configs/inbox`) et lance les configs dès leur arrivée.
```bash
PYTHONPATH=src MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
python scripts/watch_configs.py --inbox scripts/configs/inbox
```

## log_model_to_run.py
Attache un modèle Keras local (`model.keras`) à un run MLflow existant (utile si le log artefact a échoué).
```bash
PYTHONPATH=src MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
python scripts/log_model_to_run.py \
  --run-id <run_id> \
  --model-path artifacts/models/<run_name>/model.keras \
  --loss-type <ce_dice|ce_weighted|dice>
```
Assure-toi d’avoir la bonne version TF/Keras pour recharger le modèle (Keras 3).

## predict_demo.py
Démo de prédiction sur une ou plusieurs images locales. Ajuster les chemins d’entrée/sortie et le modèle à charger (local ou MLflow selon le code).
```bash
PYTHONPATH=src python scripts/predict_demo.py --help
```

## predict_runningRun.py
Prédiction en utilisant un run MLflow “running” ou récent (selon la logique du script). Passer le `run_id` ou les paramètres requis.
```bash
PYTHONPATH=src MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
python scripts/predict_runningRun.py --run-id <run_id>
```

## run_test_gpu.py
Test rapide GPU (TensorFlow) : vérifie la visibilité des devices et exécute un petit job pour valider l’accès CUDA.
```bash
python scripts/run_test_gpu.py
```
