from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import tensorflow as tf

from fvt.data.labels import (
    VOID_ID,
    remap_train_ids_to_categories,
    label_ids_to_train_ids,
)


def _find_pairs(image_dir: Path, mask_dir: Path, patterns: Iterable[str] = ("*.png", "*.jpg")) -> List[Tuple[str, str]]:
    images: List[Path] = []
    for pattern in patterns:
        images.extend(sorted(image_dir.rglob(pattern)))

    masks_lookup = _build_cityscapes_lookup(mask_dir)

    pairs: List[Tuple[str, str]] = []
    for img in images:
        key = _cityscapes_key(img.stem)
        mask = masks_lookup.get(key)
        if mask:
            pairs.append((str(img), str(mask)))
    return pairs


def _cityscapes_key(stem: str) -> str:
    """
    Normalize Cityscapes filenames to a common key (city_xxxxx_xxxxx).
    Examples:
        aachen_000000_000019_leftImg8bit -> aachen_000000_000019
        aachen_000000_000019_gtFine_labelIds -> aachen_000000_000019
    """
    parts = stem.split("_")
    return "_".join(parts[:3])


def _build_cityscapes_lookup(mask_dir: Path) -> dict:
    lookup = {}
    for p in mask_dir.rglob("*_gtFine_labelIds.png"):
        lookup[_cityscapes_key(p.stem)] = p
    # If no Cityscapes pattern found, fallback to flat stems
    if not lookup:
        lookup = {p.stem: p for p in mask_dir.iterdir() if p.is_file()}
    return lookup


def _decode_image(path: tf.Tensor) -> tf.Tensor:
    data = tf.io.read_file(path)
    image = tf.image.decode_png(data, channels=3)
    return tf.image.convert_image_dtype(image, tf.float32)


def _decode_mask(path: tf.Tensor) -> tf.Tensor:
    data = tf.io.read_file(path)
    mask = tf.image.decode_png(data, channels=1)
    mask = tf.squeeze(mask, axis=-1)
    # Convert labelIds -> trainIds if needed, then remap to 7 categories.
    mask = tf.numpy_function(label_ids_to_train_ids, [mask], tf.uint8)
    mask = tf.numpy_function(remap_train_ids_to_categories, [mask], tf.uint8)
    mask.set_shape([None, None])
    return mask


def _augment(image: tf.Tensor, mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    flip = tf.random.uniform(()) > 0.5
    image = tf.cond(flip, lambda: tf.image.flip_left_right(image), lambda: image)
    # Flip expects HxWxC; expand/squeeze to keep mask 2D afterwards.
    mask_3d = mask[..., tf.newaxis]
    mask_3d = tf.cond(flip, lambda: tf.image.flip_left_right(mask_3d), lambda: mask_3d)
    mask = tf.squeeze(mask_3d, axis=-1)
    return image, mask


def _preprocess(
    image_path: tf.Tensor,
    mask_path: tf.Tensor,
    target_size: Tuple[int, int],
    num_classes: int,
    augment: bool = False,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    image = _decode_image(image_path)
    mask = _decode_mask(mask_path)

    image = tf.image.resize(image, target_size, method="bilinear")
    mask = tf.image.resize(mask[..., tf.newaxis], target_size, method="nearest")
    mask = tf.squeeze(mask, axis=-1)

    if augment:
        image, mask = _augment(image, mask)

    valid_region = tf.cast(tf.not_equal(mask, VOID_ID), tf.float32)
    mask = tf.where(tf.equal(mask, VOID_ID), tf.zeros_like(mask), mask)

    mask = tf.one_hot(tf.cast(mask, tf.int32), depth=num_classes, dtype=tf.float32)
    sample_weight = valid_region
    return image, mask, sample_weight


def build_dataset(
    image_dir: Path,
    mask_dir: Path,
    target_size: Sequence[int] = (512, 1024),
    batch_size: int = 4,
    shuffle: bool = True,
    augment: bool = True,
    num_classes: int = 7,
) -> tf.data.Dataset:
    pairs = _find_pairs(image_dir, mask_dir)
    if not pairs:
        raise FileNotFoundError(f"No image/mask pairs found in {image_dir} and {mask_dir}.")

    image_paths, mask_paths = zip(*pairs)
    ds = tf.data.Dataset.from_tensor_slices((list(image_paths), list(mask_paths)))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(pairs), reshuffle_each_iteration=True)

    ds = ds.map(
        lambda img, msk: _preprocess(img, msk, tuple(target_size), num_classes, augment),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
