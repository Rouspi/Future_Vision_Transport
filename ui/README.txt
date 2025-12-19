# UI Streamlit

Lancement :
```bash
streamlit run ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

Fonctionnalités :
- Upload d’image.
- Visualisation de la prédiction segmentation.
- Sélection du modèle (local ou via MLflow si implémenté).

Prérequis :
- Un modèle disponible (ex. `artifacts/models/<run_name>/model.keras`) ou accessible via MLflow.
- Variable `MLFLOW_TRACKING_URI` définie si l’UI récupère un modèle via MLflow.
