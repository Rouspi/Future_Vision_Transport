import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import mlflow
import mlflow.tensorflow
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
import tensorflow as tf
import keras
import numpy as np

from fvt.config import Settings, ensure_directories
from fvt.data.dataset_builder import build_dataset
from fvt.data.labels import CATEGORY_ID_TO_NAME, palette_for_categories
from fvt.training.config import TrainingConfig
from fvt.utils.losses import DiceMetric, build_loss
from fvt.utils.models import build_model
from fvt.utils.losses import DiceMetric, build_loss
from tensorflow.keras.callbacks import Callback


def _build_backbone(input_shape: Tuple[int, int, int], trainable: bool = True) -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = trainable
    return base


def _build_unet_small(config: TrainingConfig) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=config.input_shape())
    x = inputs
    # Encoder
    c1 = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    c1 = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(p1)
    c2 = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(p2)
    c3 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    # Bottleneck
    b = tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same")(p3)
    b = tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same")(b)

    # Decoder
    u3 = tf.keras.layers.UpSampling2D((2, 2))(b)
    u3 = tf.keras.layers.Concatenate()([u3, c3])
    c4 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(u3)
    c4 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(c4)

    u2 = tf.keras.layers.UpSampling2D((2, 2))(c4)
    u2 = tf.keras.layers.Concatenate()([u2, c2])
    c5 = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(u2)
    c5 = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(c5)

    u1 = tf.keras.layers.UpSampling2D((2, 2))(c5)
    u1 = tf.keras.layers.Concatenate()([u1, c1])
    c6 = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(u1)
    c6 = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(c6)

    outputs = tf.keras.layers.Conv2D(config.num_classes, 1, activation="softmax")(c6)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="unet_small")


def _build_vgg16_unet(config: TrainingConfig) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=config.input_shape())
    x_in = tf.keras.applications.vgg16.preprocess_input(inputs)
    base = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_tensor=x_in)
    skips = [
        base.get_layer("block1_conv2").output,
        base.get_layer("block2_conv2").output,
        base.get_layer("block3_conv3").output,
        base.get_layer("block4_conv3").output,
    ]
    # Utilise le bottleneck avant le dernier pooling (block5_conv3) pour garder un stride 1/16.
    b = base.get_layer("block5_conv3").output

    # Decoder with skips
    x = tf.keras.layers.UpSampling2D((2, 2))(b)
    x = tf.keras.layers.Concatenate()([x, skips[-1]])
    x = tf.keras.layers.Conv2D(512, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2D(512, 3, activation="relu", padding="same")(x)

    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Concatenate()([x, skips[-2]])
    x = tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same")(x)

    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Concatenate()([x, skips[-3]])
    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)

    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Concatenate()([x, skips[-4]])
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)

    outputs = tf.keras.layers.Conv2D(config.num_classes, 1, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="vgg16_unet")
    return model


class TopKCheckpoints(Callback):
    """Garde tous les checkpoints si k=None, sinon les top-k sur le metric monitor."""

    def __init__(self, filepath_pattern: Path, monitor: str = "val_miou", mode: str = "max", k: Optional[int] = None):
        super().__init__()
        self.filepath_pattern = Path(filepath_pattern)
        self.monitor = monitor
        self.mode = mode
        self.k = k
        self.top: list[tuple[float, Path]] = []
        self.best_path: Optional[Path] = None

    def _better(self, a: float, b: float) -> bool:
        return a > b if self.mode == "max" else a < b

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.monitor not in logs:
            return
        metric = float(logs[self.monitor])
        path = Path(str(self.filepath_pattern).format(epoch=epoch, **{self.monitor: metric}))
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(path)
        self.top.append((metric, path))
        # Trie par metric
        self.top.sort(key=lambda x: x[0], reverse=(self.mode == "max"))
        # Best path
        self.best_path = self.top[0][1]
        # Trim top-k si demandé
        if self.k is not None and len(self.top) > self.k:
            for _, p in self.top[self.k:]:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
            self.top = self.top[: self.k]


def build_model(config: TrainingConfig) -> tf.keras.Model:
    """Factory de modèles selon config.model_type (mobilenet lite, petit UNet, VGG16 UNet)."""
    if config.model_type == "unet_small":
        model = _build_unet_small(config)
    elif config.model_type == "vgg16_unet":
        model = _build_vgg16_unet(config)
    else:
        inputs = tf.keras.Input(shape=config.input_shape())
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        backbone = _build_backbone(config.input_shape(), trainable=True)
        x = backbone(x, training=False)

        x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(x)

        x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)

        x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(x)

        logits = tf.keras.layers.Conv2D(config.num_classes, 1, padding="same")(x)
        outputs = tf.keras.layers.Softmax(name="segmentation")(logits)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_deeplab_lite")

    if config.fine_tune_from is not None:
        for layer in model.layers[: config.fine_tune_from]:
            layer.trainable = False

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    loss = build_loss(config.loss_type, config.class_weights)
    metrics = [
        tf.keras.metrics.MeanIoU(num_classes=config.num_classes, name="miou"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        DiceMetric(name="dice"),
    ]

    # We pass sample_weight to ignore void pixels; metrics are unweighted.
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        weighted_metrics=[],
    )
    return model


def _log_static_metadata(config: TrainingConfig) -> None:
    mlflow.log_params(
        {
            "input_height": config.input_height,
            "input_width": config.input_width,
            "num_classes": config.num_classes,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "augment": config.augment,
            "fine_tune_from": config.fine_tune_from,
            "mixed_precision": config.mixed_precision,
            "checkpoint_dir": str(config.checkpoint_dir),
            "class_weights": config.class_weights,
            "loss_type": config.loss_type,
            "model_type": config.model_type,
            "steps_per_epoch": config.steps_per_epoch,
            "validation_steps": config.validation_steps,
        }
    )


def _save_artifacts(model: tf.keras.Model, artifact_dir: Path) -> Dict[str, str]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # Keras 3 impose une extension explicite : on sauvegarde au format natif .keras
    saved_model_path = artifact_dir / "model.keras"
    model.save(saved_model_path)

    classes_path = artifact_dir / "classes.json"
    palette_path = artifact_dir / "palette.json"

    with classes_path.open("w") as f:
        json.dump(CATEGORY_ID_TO_NAME, f, indent=2)

    with palette_path.open("w") as f:
        json.dump(palette_for_categories(), f, indent=2)

    mlflow.log_artifacts(str(artifact_dir), artifact_path="artifacts")
    return {
        "saved_model": str(saved_model_path),
        "classes": str(classes_path),
        "palette": str(palette_path),
    }


def train_segmentation_model(config: TrainingConfig, settings: Settings) -> tf.keras.callbacks.History:
    """Entraîne un modèle de segmentation avec logging MLflow et checkpoints."""
    ensure_directories(settings)

    target_size = (config.input_height, config.input_width)
    train_ds = build_dataset(
        Path(config.train_images),
        Path(config.train_masks),
        target_size=target_size,
        batch_size=config.batch_size,
        shuffle=True,
        augment=config.augment,
        num_classes=config.num_classes,
    )
    val_ds = build_dataset(
        Path(config.val_images),
        Path(config.val_masks),
        target_size=target_size,
        batch_size=config.batch_size,
        shuffle=False,
        augment=False,
        num_classes=config.num_classes,
    )

    if config.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(config.experiment_name or settings.mlflow_experiment)

    run_suffix = config.run_name or "run"
    ckpt_dir = (settings.project_root / config.checkpoint_dir / run_suffix).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_pattern = ckpt_dir / f"{run_suffix}-{{epoch:02d}}-{{val_miou:.3f}}.weights.h5"
    ckpt_callback = TopKCheckpoints(
        filepath_pattern=ckpt_pattern,
        monitor="val_miou",
        mode="max",
        k=config.top_k_checkpoints,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_miou",
            patience=config.early_stopping_patience,
            mode="max",
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        ckpt_callback,
    ]

    model = build_model(config)

    with mlflow.start_run(run_name=config.run_name):
        print("Starting MLflow run:", config.run_name, "URI:", mlflow.get_tracking_uri())
        _log_static_metadata(config)
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.epochs,
            callbacks=callbacks,
            steps_per_epoch=config.steps_per_epoch,
            validation_steps=config.validation_steps,
        )
        # Log metrics manually to ensure miou/val_miou/dice are tracked
        for k, values in history.history.items():
            for step, v in enumerate(values):
                try:
                    mlflow.log_metric(k, float(v), step=step)
                except Exception:
                    pass

        # Recharge le meilleur checkpoint val_miou si disponible.
        best_ckpt = ckpt_callback.best_path
        if best_ckpt and best_ckpt.exists():
            try:
                model.load_weights(str(best_ckpt))
            except Exception:
                pass

        artifact_dir = (settings.project_root / config.model_output_dir / run_suffix).resolve()
        _save_artifacts(model, artifact_dir)
        signature = ModelSignature(
            inputs=Schema([TensorSpec(np.dtype("float32"), (-1,) + config.input_shape())]),
            outputs=Schema([TensorSpec(np.dtype("float32"), (-1, config.input_height, config.input_width, config.num_classes))]),
        )
        loss_for_log = build_loss(config.loss_type, config.class_weights)
        custom_objects = {"DiceMetric": DiceMetric}
        # Ajoute la loss custom si nécessaire (weighted_cce, ce_dice, dice_loss)
        if hasattr(loss_for_log, "__name__"):
            custom_objects[loss_for_log.__name__] = loss_for_log
        try:
            mlflow.keras.log_model(
                model,
                artifact_path="model",
                signature=signature,
                pip_requirements=[
                    f"tensorflow=={tf.__version__}",
                    f"keras=={keras.__version__}",
                    "cloudpickle",
                ],
                custom_objects=custom_objects,
            )
        except TypeError:
            # Compatibilité MLflow sans support custom_objects (on log sans ces objets).
            mlflow.keras.log_model(
                model,
                artifact_path="model",
                signature=signature,
                pip_requirements=[
                    f"tensorflow=={tf.__version__}",
                    f"keras=={keras.__version__}",
                    "cloudpickle",
                ],
            )

        # Log des checkpoints dans MLflow (tous ou top-k selon config.top_k_checkpoints).
        try:
            mlflow.log_artifacts(str(ckpt_dir), artifact_path="checkpoints")
        except Exception:
            pass

    return history
