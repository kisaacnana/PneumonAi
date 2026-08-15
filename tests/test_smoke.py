import tensorflow as tf

from Code.train_modern import IMG_SIZE, build_model


def test_model_builds_with_expected_input_shape():
    model = build_model()
    assert model.input_shape == (None, IMG_SIZE[0], IMG_SIZE[1], 3)
    assert model.output_shape == (None, 1)


def test_model_outputs_probability():
    model = build_model()
    batch = tf.zeros((2, IMG_SIZE[0], IMG_SIZE[1], 3))
    predictions = model(batch, training=False)
    assert predictions.shape == (2, 1)
    assert tf.reduce_all(predictions >= 0)
    assert tf.reduce_all(predictions <= 1)
