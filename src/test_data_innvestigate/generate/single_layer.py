"""
Run all LRP Analyzers over single layer models
using random weights and random input.
"""
import os
from pathlib import Path

import h5py
import keras
import numpy as np

from test_data_innvestigate.utils.analyzers import METHODS

ROOT_DIR = os.path.abspath(os.curdir)

# Create input data representative of a single 2D image with 3 channels
# Using the default `data_format="channels_last"`:
#    (batch_size, rows, cols, channels)
INPUT_SHAPE = (10, 10, 3)
BATCH_SIZE = 1

# Layers used in unit tests
KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)

LAYERS_2D = {
    "Dense": keras.layers.Dense(5, input_shape=INPUT_SHAPE),
    "Dense_relu": keras.layers.Dense(5, activation="relu", input_shape=INPUT_SHAPE),
    "Conv2D": keras.layers.convolutional.Conv2D(
        5, KERNEL_SIZE, input_shape=INPUT_SHAPE
    ),
    "Conv2D_relu": keras.layers.convolutional.Conv2D(
        5, KERNEL_SIZE, activation="relu", input_shape=INPUT_SHAPE
    ),
    "AveragePooling2D": keras.layers.pooling.AveragePooling2D(
        POOL_SIZE, input_shape=INPUT_SHAPE
    ),
    "MaxPooling2D": keras.layers.pooling.MaxPooling2D(
        POOL_SIZE, input_shape=INPUT_SHAPE
    ),
}


def get_single_layer_model(layer):
    """Return Keras model consisting of `layer`."""
    weights = layer.get_weights()
    model = keras.Sequential()
    model.add(layer)
    model.add(keras.layers.Flatten())

    return model, weights


def generate():
    """Run random input through single layers, applying all LRP Analyzers."""
    print("Generating data on layers...")
    x = np.random.rand(BATCH_SIZE, *INPUT_SHAPE)

    # Create data folder
    Path(os.path.join(ROOT_DIR, "data", "layer")).mkdir(parents=True, exist_ok=True)

    for analyzer_name, m in METHODS.items():
        print("\t... using {}".format(analyzer_name))
        method, kwargs = m

        # Write to hdf5 file
        data_path = os.path.join(ROOT_DIR, "data", "layer", analyzer_name + ".hdf5")
        with h5py.File(data_path, "w") as f:

            f.create_dataset("input", data=x)
            f.attrs["analyzer_name"] = analyzer_name

            for layer_name, layer in LAYERS_2D.items():
                # Get model
                model, weights = get_single_layer_model(layer)
                # Analyze model
                analyzer = method(model, **kwargs)
                a = analyzer.analyze(x)
                y = model.predict(x)

                # Save attribution
                l = f.create_group(layer_name)
                l.attrs["layer_name"] = layer_name
                l.create_dataset("attribution", data=a)
                l.create_dataset("output", data=y)

                # Save layer weights
                ws = l.create_group("weights")
                for i, weight in enumerate(weights):
                    ws.create_dataset(str(i), data=weight)


if __name__ == "__main__":
    generate()
