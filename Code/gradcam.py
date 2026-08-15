"""Generate a Grad-CAM heatmap for a PneumonAi prediction.

Grad-CAM highlights image regions that contribute to a CNN prediction. It is
an interpretability aid, not proof that the model is using clinically valid
features.
"""

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = (224, 224)
MODEL_PATH = Path("artifacts/pneumonai.keras")
IMAGE_PATH = Path("sample_xray.png")
OUTPUT_PATH = Path("artifacts/gradcam.png")


def find_last_conv_layer(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the model.")


def make_gradcam(model, image: tf.Tensor, layer_name: str):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, prediction = grad_model(image)
        score = prediction[:, 0]

    gradients = tape.gradient(score, conv_outputs)
    weights = tf.reduce_mean(gradients, axis=(1, 2))
    cam = tf.reduce_sum(conv_outputs * weights[:, None, None, :], axis=-1)
    cam = tf.maximum(cam, 0)
    cam = cam / (tf.reduce_max(cam, axis=(1, 2), keepdims=True) + 1e-8)
    return cam[0].numpy(), float(prediction[0, 0].numpy())


def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Place a chest X-ray at {IMAGE_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)
    image = tf.keras.utils.load_img(IMAGE_PATH, target_size=IMG_SIZE)
    array = tf.keras.utils.img_to_array(image)
    tensor = tf.expand_dims(array, 0)

    layer_name = find_last_conv_layer(model)
    heatmap, probability = make_gradcam(model, tensor, layer_name)

    original = cv2.imread(str(IMAGE_PATH))
    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_PATH), overlay)

    label = "PNEUMONIA" if probability >= 0.5 else "NORMAL"
    print(f"Prediction: {label}")
    print(f"Pneumonia probability: {probability:.4f}")
    print(f"Grad-CAM layer: {layer_name}")
    print(f"Saved explanation to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
