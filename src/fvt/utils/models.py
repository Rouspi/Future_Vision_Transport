from typing import Tuple

import tensorflow as tf

from fvt.training.config import TrainingConfig


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
    b = base.get_layer("block5_conv3").output

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

    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)

    outputs = tf.keras.layers.Conv2D(config.num_classes, 1, activation="softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="vgg16_unet")


def build_model(config: TrainingConfig) -> tf.keras.Model:
    """Factory selon config.model_type."""
    if config.model_type == "unet_small":
        return _build_unet_small(config)
    if config.model_type == "vgg16_unet":
        return _build_vgg16_unet(config)

    # MobileNetV2 lite head (par défaut)
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
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_deeplab_lite")
