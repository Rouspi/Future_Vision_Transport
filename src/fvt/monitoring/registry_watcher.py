import time
from pathlib import Path
from typing import Optional

import mlflow
import typer

from fvt.config import load_settings
from fvt.monitoring.mlflow_tracking import configure_mlflow

app = typer.Typer(add_completion=False)


def _download_production_model(
    model_name: str,
    stage: str,
    dst_path: Path,
) -> Optional[str]:
    settings = load_settings()
    client = configure_mlflow(settings, experiment=settings.mlflow_experiment)
    versions = client.get_latest_versions(model_name, stages=[stage])
    if not versions:
        return None

    uri = f"models:/{model_name}/{stage}"
    dst_path.mkdir(parents=True, exist_ok=True)
    local_path = mlflow.artifacts.download_artifacts(uri=uri, dst_path=str(dst_path))
    return local_path


@app.command()
def pull(
    model_name: str = "future_vision_segmentation",
    stage: str = "Production",
    dst: Path = Path("artifacts/production_bundle"),
) -> None:
    """Pull the latest model for a given stage from the MLflow registry."""
    result = _download_production_model(model_name, stage, dst)
    if result is None:
        typer.echo(f"Aucune version {stage} pour {model_name}.")
        raise typer.Exit(code=1)
    typer.echo(f"Téléchargé {model_name} ({stage}) -> {result}")


@app.command()
def watch(
    model_name: str = "future_vision_segmentation",
    stage: str = "Production",
    dst: Path = Path("artifacts/production_bundle"),
    interval: int = 600,
) -> None:
    """Poll the registry and download a new Production model when it changes."""
    last_version: Optional[str] = None
    while True:
        settings = load_settings()
        client = configure_mlflow(settings, experiment=settings.mlflow_experiment)
        versions = client.get_latest_versions(model_name, stages=[stage])
        if versions:
            version = versions[0]
            if version.version != last_version:
                path = _download_production_model(model_name, stage, dst)
                if path:
                    typer.echo(f"Nouveau modèle {model_name} v{version.version} ({stage}) -> {path}")
                    last_version = version.version
        time.sleep(interval)


if __name__ == "__main__":
    app()
