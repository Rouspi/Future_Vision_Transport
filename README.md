# Future Vision Transport

Ce dépôt contient le code d’entraînement et d’inférence pour un projet de segmentation d’images (Cityscapes).

## Structure
- `scripts/` : lancement d’expériences (run_configs.py, configs TOML), utilitaires (log_model_to_run.py).
- `src/fvt/` : librairie principale (données, modèles, pertes, pipeline d’entraînement, métriques, utils MLflow).
- `api/` : API FastAPI pour l’inférence.
- `ui/` : interface Streamlit de démonstration.
- `notebooks/` : explorations et visualisations.
- `mlruns/` : tracking MLflow (si lancé en local).
- `artifacts/` : modèles et checkpoints sauvegardés localement (suffixés par le nom du run).

## Installation rapide
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# + pip install -e .  # pour installer la librairie fvt en editable
```

## Entraîner un ou plusieurs modèles
Placer vos configs TOML dans `scripts/configs/inbox/`, puis :
```bash
PYTHONPATH=src MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
python scripts/run_configs.py --inbox scripts/configs/inbox
```
Les configs sont déplacées en `processed/` ou `failed/` selon le résultat.

## API FastAPI
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

## UI Streamlit
```bash
streamlit run ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

## MLflow en local
```bash
mlflow ui \
  --backend-store-uri file:./mlruns \
  --default-artifact-root file:./mlruns \
  --host 127.0.0.1 --port 5000
```

## Notes plateforme
- TensorFlow 2.17–<2.20 (GPU ou CPU) est visé. Sur Mac Intel, pip ne propose pas >2.16.2 : dans ce cas, utiliser `tensorflow==2.16.2` (et éventuellement `tensorflow-macos` sur Apple Silicon).
- Pour charger les modèles sauvegardés en Keras 3, assurez-vous d’avoir une version TF/Keras compatible (ex. TF 2.16.2 + Keras 3.4.x sur macOS).
