"""Evaluate a trained PneumonAi model on the test set."""

from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_ROOT = Path("data/chest_xray")
MODEL_PATH = Path("artifacts/pneumonai.keras")


def main() -> None:
    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_ROOT / "test",
        labels="inferred",
        label_mode="binary",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = tf.keras.models.load_model(MODEL_PATH)
    probabilities = model.predict(test_ds, verbose=0).ravel()
    y_true = np.concatenate([labels.numpy().ravel() for _, labels in test_ds]).astype(int)
    y_pred = (probabilities >= 0.5).astype(int)

    print(classification_report(y_true, y_pred, target_names=["NORMAL", "PNEUMONIA"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_true, probabilities):.4f}")


if __name__ == "__main__":
    main()
