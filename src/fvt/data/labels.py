from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

VOID_ID = 255


@dataclass(frozen=True)
class Label:
    name: str
    train_id: int
    category: str
    category_id: int
    color: Tuple[int, int, int]


CATEGORY_NAME_TO_ID: Dict[str, int] = {
    "flat": 0,
    "construction": 1,
    "object": 2,
    "nature": 3,
    "sky": 4,
    "human": 5,
    "vehicle": 6,
}

CATEGORY_ID_TO_NAME: Dict[int, str] = {v: k for k, v in CATEGORY_NAME_TO_ID.items()}

# Cityscapes train IDs (19 classes) mapped to the 8 main categories (void is ignored via VOID_ID).
CITYSCAPES_LABELS: List[Label] = [
    Label("road", 0, "flat", CATEGORY_NAME_TO_ID["flat"], (128, 64, 128)),
    Label("sidewalk", 1, "flat", CATEGORY_NAME_TO_ID["flat"], (244, 35, 232)),
    Label("building", 2, "construction", CATEGORY_NAME_TO_ID["construction"], (70, 70, 70)),
    Label("wall", 3, "construction", CATEGORY_NAME_TO_ID["construction"], (102, 102, 156)),
    Label("fence", 4, "construction", CATEGORY_NAME_TO_ID["construction"], (190, 153, 153)),
    Label("pole", 5, "object", CATEGORY_NAME_TO_ID["object"], (153, 153, 153)),
    Label("traffic light", 6, "object", CATEGORY_NAME_TO_ID["object"], (250, 170, 30)),
    Label("traffic sign", 7, "object", CATEGORY_NAME_TO_ID["object"], (220, 220, 0)),
    Label("vegetation", 8, "nature", CATEGORY_NAME_TO_ID["nature"], (107, 142, 35)),
    Label("terrain", 9, "nature", CATEGORY_NAME_TO_ID["nature"], (152, 251, 152)),
    Label("sky", 10, "sky", CATEGORY_NAME_TO_ID["sky"], (70, 130, 180)),
    Label("person", 11, "human", CATEGORY_NAME_TO_ID["human"], (220, 20, 60)),
    Label("rider", 12, "human", CATEGORY_NAME_TO_ID["human"], (255, 0, 0)),
    Label("car", 13, "vehicle", CATEGORY_NAME_TO_ID["vehicle"], (0, 0, 142)),
    Label("truck", 14, "vehicle", CATEGORY_NAME_TO_ID["vehicle"], (0, 0, 70)),
    Label("bus", 15, "vehicle", CATEGORY_NAME_TO_ID["vehicle"], (0, 60, 100)),
    Label("train", 16, "vehicle", CATEGORY_NAME_TO_ID["vehicle"], (0, 80, 100)),
    Label("motorcycle", 17, "vehicle", CATEGORY_NAME_TO_ID["vehicle"], (0, 0, 230)),
    Label("bicycle", 18, "vehicle", CATEGORY_NAME_TO_ID["vehicle"], (119, 11, 32)),
]

TRAIN_ID_TO_LABEL: Dict[int, Label] = {label.train_id: label for label in CITYSCAPES_LABELS}
TRAIN_ID_TO_CATEGORY_ID: Dict[int, int] = {
    label.train_id: label.category_id for label in CITYSCAPES_LABELS
}

# Cityscapes official mapping from labelIds (0-33, 255) to trainIds (0-18, 255)
# https://www.cityscapes-dataset.com/dataset-overview/#labels
_LABELID_TO_TRAINID_ARRAY = np.full(256, VOID_ID, dtype=np.uint8)
for label_id, train_id in [
    (0, 255),  # unlabeled
    (1, 255),  # ego vehicle
    (2, 255),  # rectification border
    (3, 255),  # out of roi
    (4, 255),  # static
    (5, 255),  # dynamic
    (6, 255),  # ground
    (7, 0),  # road
    (8, 1),  # sidewalk
    (9, 255),  # parking
    (10, 255),  # rail track
    (11, 2),  # building
    (12, 3),  # wall
    (13, 4),  # fence
    (14, 255),  # guard rail
    (15, 255),  # bridge
    (16, 255),  # tunnel
    (17, 5),  # pole
    (18, 255),  # polegroup
    (19, 6),  # traffic light
    (20, 7),  # traffic sign
    (21, 8),  # vegetation
    (22, 9),  # terrain
    (23, 10),  # sky
    (24, 11),  # person
    (25, 12),  # rider
    (26, 13),  # car
    (27, 14),  # truck
    (28, 15),  # bus
    (29, 255),  # caravan
    (30, 255),  # trailer
    (31, 16),  # train
    (32, 17),  # motorcycle
    (33, 18),  # bicycle
    (255, 255),  # void
]:
    _LABELID_TO_TRAINID_ARRAY[label_id] = np.uint8(train_id)


def remap_train_ids_to_categories(mask: np.ndarray) -> np.ndarray:
    """Map Cityscapes train IDs to the 8 coarse categories (void stays VOID_ID)."""
    if mask.ndim != 2:
        raise ValueError("Mask must be 2D HxW with train IDs.")

    remapped = np.full_like(mask, VOID_ID, dtype=np.uint8)
    for train_id, category_id in TRAIN_ID_TO_CATEGORY_ID.items():
        remapped[mask == np.uint8(train_id)] = np.uint8(category_id)
    return remapped


def palette_for_categories() -> List[Tuple[int, int, int]]:
    """Return a 7-color palette (void handled separately)."""
    return [
        (128, 64, 128),  # flat
        (70, 70, 70),  # construction
        (220, 220, 0),  # object
        (107, 142, 35),  # nature
        (70, 130, 180),  # sky
        (220, 20, 60),  # human
        (0, 0, 142),  # vehicle
    ]


def colorize(mask: np.ndarray, palette: Iterable[Tuple[int, int, int]] | None = None) -> np.ndarray:
    """Convert a category mask (HxW) into an RGB image using the palette."""
    if mask.ndim != 2:
        raise ValueError("Mask must be 2D HxW.")

    palette_values = list(palette) if palette is not None else palette_for_categories()
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for idx, color in enumerate(palette_values):
        rgb[mask == idx] = color
    return rgb


def label_ids_to_train_ids(mask: np.ndarray) -> np.ndarray:
    """Convert Cityscapes labelIds to trainIds (unsupported ids become VOID_ID)."""
    if mask.ndim != 2:
        raise ValueError("Mask must be 2D HxW.")
    return _LABELID_TO_TRAINID_ARRAY[mask.astype(np.uint8)]


__all__ = [
    "VOID_ID",
    "Label",
    "CITYSCAPES_LABELS",
    "TRAIN_ID_TO_LABEL",
    "TRAIN_ID_TO_CATEGORY_ID",
    "remap_train_ids_to_categories",
    "palette_for_categories",
    "colorize",
    "label_ids_to_train_ids",
]
