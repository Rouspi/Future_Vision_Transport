from typing import Optional, Tuple

import tensorflow as tf


def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """Dice coefficient moyen sur les classes (y_true/y_pred one-hot)."""
    y_true_f = tf.reshape(y_true, (tf.shape(y_true)[0], -1, tf.shape(y_true)[-1]))
    y_pred_f = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1, tf.shape(y_pred)[-1]))
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    denom = tf.reduce_sum(y_true_f + y_pred_f, axis=1)
    dice_per_class = (2.0 * intersection + smooth) / (denom + smooth)
    return tf.reduce_mean(dice_per_class, axis=-1)


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return 1.0 - tf.reduce_mean(dice_coefficient(y_true, y_pred))


class DiceMetric(tf.keras.metrics.Mean):
    """Average dice coefficient over batch."""

    def __init__(self, name: str = "dice", **kwargs):
        super().__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        dice = dice_coefficient(y_true, y_pred)
        return super().update_state(dice, sample_weight=sample_weight)


def build_loss(loss_type: str, class_weights: Optional[Tuple[float, ...]] = None) -> tf.keras.losses.Loss:
    """Factory for losses: weighted CE, Dice, or CE+Dice."""

    def weighted_cce(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        if class_weights:
            class_w = tf.constant(class_weights, dtype=tf.float32)
            weights = tf.reduce_sum(class_w * y_true, axis=-1)
        else:
            weights = 1.0
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        return ce * weights

    if loss_type == "dice":
        return dice_loss
    if loss_type == "ce_dice":
        def ce_dice(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            return weighted_cce(y_true, y_pred) + dice_loss(y_true, y_pred)
        return ce_dice

    return weighted_cce
