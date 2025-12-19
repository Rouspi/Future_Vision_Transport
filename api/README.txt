# API FastAPI

Endpoints principaux :
- POST /predict : accepte une image et retourne le masque segmenté.
- GET /health : ping.

Lancement local :
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Variables utiles :
- `MLFLOW_TRACKING_URI` : si l’API doit charger un modèle depuis MLflow.
- `MODEL_PATH` : chemin local vers un `model.keras` (optionnel selon l’implémentation).

Dépendances : voir `requirements.txt` (fastapi, uvicorn, pydantic, python-multipart, httpx/requests).
