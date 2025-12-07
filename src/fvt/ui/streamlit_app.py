import base64
import io
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import requests
import streamlit as st
from PIL import Image

from fvt.config import load_settings
from fvt.data.labels import colorize, remap_train_ids_to_categories, label_ids_to_train_ids

settings = load_settings()

DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
DEFAULT_IMAGE_DIR = Path(
    os.getenv(
        "DEMO_IMAGE_DIR",
        settings.data_root
        / "downloads"
        / "P8_Cityscapes_leftImg8bit_trainvaltest"
        / "leftImg8bit"
        / "val",
    )
)
DEFAULT_MASK_DIR = Path(
    os.getenv(
        "DEMO_MASK_DIR",
        settings.data_root
        / "downloads"
        / "P8_Cityscapes_gtFine_trainvaltest"
        / "gtFine"
        / "val",
    )
)


def list_images(image_dir: Path) -> List[Path]:
    """List images recursively to include Cityscapes subfolders."""
    return sorted(
        [p for p in image_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )


def find_cityscapes_mask(image_path: Path, image_root: Path, mask_root: Path) -> Optional[Path]:
    """
    Cityscapes masks live under gtFine/<split>/<city>/..._gtFine_labelTrainIds.png
    while images are under leftImg8bit/<split>/<city>/..._leftImg8bit.png.
    We try trainIds first, then fallback to labelIds.
    """
    try:
        rel = image_path.relative_to(image_root)
    except ValueError:
        rel = image_path.name
    stems = [
        image_path.stem.replace("_leftImg8bit", "_gtFine_labelTrainIds"),
        image_path.stem.replace("_leftImg8bit", "_gtFine_labelIds"),
    ]
    for stem in stems:
        candidate = (mask_root / Path(rel).parent / stem).with_suffix(".png")
        if candidate.exists():
            return candidate
    return None


def decode_mask(b64_mask: str) -> Image.Image:
    data = base64.b64decode(b64_mask.encode("ascii"))
    return Image.open(io.BytesIO(data))


def main() -> None:
    st.title("Future Vision Transport - Demo segmentation")
    st.markdown("Teste l'API FastAPI, affiche l'image, le mask réel (si dispo) et le mask prédit.")

    api_url = st.sidebar.text_input("API URL", DEFAULT_API_URL)
    image_dir = Path(st.sidebar.text_input("Répertoire images", str(DEFAULT_IMAGE_DIR)))
    mask_dir = Path(st.sidebar.text_input("Répertoire masks (optionnel)", str(DEFAULT_MASK_DIR)))

    images = list_images(image_dir)
    if not images:
        st.warning(f"Aucune image trouvée dans {image_dir}")
        return

    selected = st.selectbox("Image", options=[str(p.relative_to(image_dir)) for p in images])
    image_path = image_dir / selected
    mask_path = (
        find_cityscapes_mask(image_path, image_dir, mask_dir) if mask_dir.exists() else None
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(Image.open(image_path), caption="Image", use_column_width=True)
    if mask_path and mask_path.exists():
        with col2:
            mask_arr = np.array(Image.open(mask_path))
            if mask_arr.ndim == 2:
                # Convert labelIds -> trainIds if needed, then remap to 7 categories.
                if mask_arr.max() > 18:  # likely labelIds
                    mask_arr = label_ids_to_train_ids(mask_arr)
                mask_arr = remap_train_ids_to_categories(mask_arr)
                mask_arr = colorize(mask_arr)
            st.image(mask_arr, caption="Mask réel", use_column_width=True)

    if st.button("Lancer la prédiction"):
        with image_path.open("rb") as f:
            files = {"file": (image_path.name, f, "image/png")}
            try:
                response = requests.post(api_url, files=files, timeout=60)
            except requests.RequestException as exc:
                st.error(f"Appel API impossible: {exc}")
                return

        if response.status_code != 200:
            st.error(f"Erreur API {response.status_code}: {response.text}")
            return

        payload = response.json()
        mask_img = decode_mask(payload["mask_base64"])

        with col3:
            st.image(mask_img, caption="Mask prédit", use_column_width=True)

        st.json(
            {
                "latency_ms": payload.get("latency_ms"),
                "model_name": payload.get("model_name"),
                "model_stage": payload.get("model_stage"),
                "payload_hash": payload.get("payload_hash"),
            }
        )


if __name__ == "__main__":
    main()
