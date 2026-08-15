"""Train a reproducible CNN pneumonia classifier.

This script expects the Kaggle chest X-ray dataset to be arranged as:

data/chest_xray/train/NORMAL
 data/chest_xray/train/PNEUMONIA
 data/chest_xray/val/NORMAL
 data/chest_xray/val/PNEUMONIA
 data/chest_xray/test/NORMAL
 data/chest_xray/test/PNEUMONIA

The dataset is intentionally not committed to this repository.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
DATA_ROOT = Path("data/chest_xray")
OUTPUT_DIR = Path("artifacts")


def load_split(name: str, shuffle: bool):
    return keras.utils.image_dataset_from_directory(
        DATA_ROOT / name,
        labels="inferred",
        label_mode="binary",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )


def build_model() -> keras.Model:
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.1)(x)

    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="pneumonai_cnn")


def plot_history(history: keras.callbacks.History) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("PneumonAi Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accuracy.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PneumonAi Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss.png", dpi=160)
    plt.close()


def main() -> None:
    tf.keras.utils.set_random_seed(SEED)

    train_ds = load_split("train", shuffle=True)
    val_ds = load_split("val", shuffle=False)
    test_ds = load_split("test", shuffle=False)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=4, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            OUTPUT_DIR / "pneumonai.keras", monitor="val_auc", mode="max", save_best_only=True
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    results = model.evaluate(test_ds, return_dict=True)
    print("Test metrics:")
    for name, value in results.items():
        print(f"  {name}: {value:.4f}")

    plot_history(history)


if __name__ == "__main__":
    main()
