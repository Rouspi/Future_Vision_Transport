from dataclasses import dataclass
from pathlib import Path
import os


DEFAULT_EXPERIMENT_NAME = "future_vision_segmentation"


@dataclass
class Settings:
    """Container for project-wide paths and MLflow configuration."""

    project_root: Path
    data_root: Path
    downloads_dir: Path
    raw_dir: Path
    processed_dir: Path
    interim_dir: Path
    mlruns_dir: Path
    mlflow_tracking_uri: str
    mlflow_experiment: str


def load_settings() -> Settings:
    repo_root = Path(__file__).resolve().parents[2]
    data_root = repo_root / "data"

    mlruns_dir = repo_root / "mlruns"
    # Default to external MLflow server if provided; fallback to local file-based backend.
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    if tracking_uri == "http://localhost:5000" and not os.getenv("MLFLOW_TRACKING_URI"):
        # If the local server is not reachable, MLflow will transparently create the dir backend.
        tracking_uri = os.getenv("MLFLOW_FALLBACK_URI", str(mlruns_dir))
    experiment = os.getenv("MLFLOW_EXPERIMENT", DEFAULT_EXPERIMENT_NAME)

    return Settings(
        project_root=repo_root,
        data_root=data_root,
        downloads_dir=data_root / "downloads",
        raw_dir=data_root / "raw",
        processed_dir=data_root / "processed",
        interim_dir=data_root / "interim",
        mlruns_dir=mlruns_dir,
        mlflow_tracking_uri=tracking_uri,
        mlflow_experiment=experiment,
    )


def ensure_directories(settings: Settings) -> None:
    """Create common project directories if they are missing."""
    for path in (
        settings.data_root,
        settings.downloads_dir,
        settings.raw_dir,
        settings.processed_dir,
        settings.interim_dir,
        settings.mlruns_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
