import io
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from api.app import app

client = TestClient(app)


def _fake_image_bytes(width=1024, height=512):
    arr = (np.random.rand(height, width, 3) * 255).astype("uint8")
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_returns_mask():
    files = {"file": ("fake.png", _fake_image_bytes(), "image/png")}
    r = client.post("/predict", files=files)
    assert r.status_code == 200
    data = r.json()
    # On vérifie les champs clés sans valider le contenu du masque
    assert "mask_base64" in data
    assert data["height"] == 512
    assert data["width"] == 1024
    assert data["model_source"]  # local ou registry
