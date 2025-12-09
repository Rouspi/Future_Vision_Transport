from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class TrainingConfig:
    input_height: int = 512
    input_width: int = 1024
    num_classes: int = 7  # 8th category is void and ignored via sample weights
    batch_size: int = 2
    epochs: int = 20
    steps_per_epoch: Optional[int] = None
    validation_steps: Optional[int] = None
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    augment: bool = True
    early_stopping_patience: int = 6
    model_output_dir: Path = Path("artifacts/models")
    checkpoint_dir: Path = Path("artifacts/checkpoints")
    experiment_name: str = "future_vision_segmentation"
    run_name: Optional[str] = None
    fine_tune_from: Optional[int] = None  # freeze backbone until this layer index
    mixed_precision: bool = False
    # Loss/architecture
    loss_type: str = "ce_weighted"  # options: ce_weighted, dice, ce_dice
    model_type: str = "mobilenetv2_deeplab_lite"  # options: mobilenetv2_deeplab_lite, unet_small, vgg16_unet
    # Optional class weights (order matches categories in labels.py palette)
    # Example: give more weight to humans/vehicles (classes rares) to improve recall.
    class_weights: Optional[Tuple[float, ...]] = (
        1.0,  # flat
        1.0,  # construction
        1.5,  # object (signs, lights)
        1.0,  # nature
        1.0,  # sky
        2.0,  # human
        2.0,  # vehicle
    )

    # Defaults pointing directly to Cityscapes download layout
    train_images: Path = Path("data/downloads/P8_Cityscapes_leftImg8bit_trainvaltest/leftImg8bit/train")
    train_masks: Path = Path("data/downloads/P8_Cityscapes_gtFine_trainvaltest/gtFine/train")
    val_images: Path = Path("data/downloads/P8_Cityscapes_leftImg8bit_trainvaltest/leftImg8bit/val")
    val_masks: Path = Path("data/downloads/P8_Cityscapes_gtFine_trainvaltest/gtFine/val")

    # Checkpoints : None => on garde tous les checkpoints, sinon on garde les top-K val_miou.
    top_k_checkpoints: Optional[int] = None

    def input_shape(self) -> Tuple[int, int, int]:
        return (self.input_height, self.input_width, 3)
