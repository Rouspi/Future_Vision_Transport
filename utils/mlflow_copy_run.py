
PYTHONPATH=src
MLFLOW_TRACKING_URI=http://localhost:5000

import shutil, tempfile
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

SOURCE_RUN = "3b9679dfb26e4906874b9ba7277c54ca"
TARGET_RUN = "7a158687a99a44b984ffcda0e5ee2426"

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

# Copier params/tags du run source vers le run cible (sans écraser ce qui existe déjà)
src = client.get_run(SOURCE_RUN)
for k,v in src.data.params.items():
    client.log_param(TARGET_RUN, k, v)
for k,v in src.data.tags.items():
    if k not in {"mlflow.runName","mlflow.source.name","mlflow.source.git.commit"}:
        client.set_tag(TARGET_RUN, k, v)

# Copier artefacts
with tempfile.TemporaryDirectory() as tmpdir:
    local_dir = client.download_artifacts(SOURCE_RUN, "", tmpdir)
    for p in Path(local_dir).rglob("*"):
        if p.is_file():
            rel = p.relative_to(local_dir)
            client.log_artifact(TARGET_RUN, str(p), artifact_path=str(rel.parent) if rel.parent != Path(".") else None)

print("Copie terminée.")

