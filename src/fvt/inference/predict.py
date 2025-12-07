from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

from fvt.data.labels import colorize, palette_for_categories


@dataclass
class InferenceConfig:
    target_height: int = 512
    target_width: int = 1024
    num_classes: int = 7

    def target_size(self) -> Tuple[int, int]:
        return (self.target_height, self.target_width)


def load_model(model_path: Path) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)


def preprocess_image(image_path: Path, target_size: Tuple[int, int]) -> tf.Tensor:
    data = tf.io.read_file(str(image_path))
    image = tf.image.decode_image(data, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_size, method="bilinear")
    return image


def predict_mask(
    model: tf.keras.Model,
    image_tensor: tf.Tensor,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    logits = model(image_tensor[tf.newaxis, ...], training=False)
    probs = tf.nn.softmax(logits, axis=-1)
    probs_np = probs.numpy()[0]
    mask = np.argmax(probs_np, axis=-1).astype(np.uint8)
    return mask, probs_np


def mask_to_png_bytes(mask: np.ndarray) -> bytes:
    rgb = colorize(mask, palette_for_categories())
    image = Image.fromarray(rgb)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.read()


def mask_to_base64(mask: np.ndarray) -> str:
    png_bytes = mask_to_png_bytes(mask)
    return base64.b64encode(png_bytes).decode("ascii")
