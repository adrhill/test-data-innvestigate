import os
from pathlib import Path

import h5py

import innvestigate.analyzer.relevance_based.relevance_rule as rrule

import keras
import keras.backend as K

import numpy as np

ROOT_DIR = os.path.abspath(os.curdir)
LRP_RULES = [
    "ZRule",
    "ZIgnoreBiasRule",
    "EpsilonRule",
    "EpsilonIgnoreBiasRule",
    "WSquareRule",
    "FlatRule",
    "Alpha2Beta1Rule",
    "Alpha2Beta1IgnoreBiasRule",
    "Alpha1Beta0Rule",
    "Alpha1Beta0IgnoreBiasRule",
    "AlphaBetaXRule",
    "AlphaBetaX1000Rule",
    "AlphaBetaX1010Rule",
    "AlphaBetaX1001Rule",
    "AlphaBetaX2m100Rule",
    "ZPlusRule",
    "ZPlusFastRule",
    "BoundedRule",
]

# Create input data representative of a single 2D image with 3 channels
# Using the default `data_format="channels_last"`:
#    (batch_size, rows, cols, channels)
INPUT_SHAPE = (10, 10, 3)
BATCH_SIZE = 1


def get_single_layer_model(layer):
    x = keras.layers.Input(shape=INPUT_SHAPE)
    y = layer(x)

    model = keras.Model(x, y)
    weights = layer.get_weights()

    return model, weights


def run_rule(layer, layer_name: str, x: np.ndarray):
    print("Generating data on layer {}".format(layer_name))
    K.clear_session()

    # Create data folder
    Path(os.path.join(ROOT_DIR, "data", "layer")).mkdir(parents=True, exist_ok=True)

    # Get model
    model, weights = get_single_layer_model(layer)

    # Get prediction
    X = K.constant(x)  # ndarray to tensor
    y = model.predict(X, steps=BATCH_SIZE)  # TODO: obtain tensor
    # y = K.eval(Y) # tensor to ndarray
    Y = K.constant(y)  # ndarray to tensor
    U = K.ones_like(Y)  # uniform ones

    print("\tinput_dim: {}".format(np.shape(x)))
    print("\toutput_dim: {}".format(np.shape(y)))

    # Create hdf5 data
    data_path = os.path.join(ROOT_DIR, "data", "layer", layer_name + ".hdf5")

    with h5py.File(data_path, "w") as f:
        f.attrs["layer_name"] = layer_name
        f.create_dataset("input", data=x)
        f.create_dataset("output", data=y)
        attribs = f.create_group("attributions")

        # Apply all rules on ys and uniform output
        for rule_name in LRP_RULES:
            # state and reverse_state are required to
            # construct and apply LRP-rules respectively
            state = None
            reverse_state = None

            # Create ReverseAnalyzerBase using rule
            Rule = getattr(rrule, rule_name)  # get class of rule
            rule = Rule(layer, state)
            print("\t... using {}: {}".format(rule_name, rule))

            # Get rule matching name from iNNvestigate relevance_rules
            r = attribs.create_group(rule_name)

            # Calculate relevances for output Rs=ys of layers
            rel_out = rule.apply([X], [Y], [Y], reverse_state)
            rel_out = K.eval(rel_out[0])  # tensor in singleton list to numpy array
            r.create_dataset("output", data=rel_out)

            # Calculate relevances for uniform Rs
            rel_unif = rule.apply([X], [Y], [U], reverse_state)  # TODO: keep xs?
            rel_unif = K.eval(rel_unif[0])  # tensor in singleton list to numpy array
            r.create_dataset("uniform", data=rel_unif)


def generate():
    """Create single layer model and run LRP"""

    xs = np.random.rand(BATCH_SIZE, *INPUT_SHAPE)

    # Create layers that are evaluated
    kernel_size = (3, 3)
    # pool_size = (2, 2)

    layers_2D = {
        "Dense": keras.layers.Dense(5, input_shape=INPUT_SHAPE),
        "Dense_relu": keras.layers.Dense(5, activation="relu", input_shape=INPUT_SHAPE),
        "Conv2D": keras.layers.convolutional.Conv2D(
            5, kernel_size, input_shape=INPUT_SHAPE
        ),
        "Conv2D_relu": keras.layers.convolutional.Conv2D(
            5, kernel_size, activation="relu", input_shape=INPUT_SHAPE
        ),
        # "AveragePooling2D": keras.layers.pooling.AveragePooling2D(
        #     pool_size, input_shape=INPUT_SHAPE
        # ),
        # "MaxPooling2D": keras.layers.pooling.MaxPooling2D(
        #     pool_size, input_shape=INPUT_SHAPE
        # ),
        # "GlobalAveragePooling2D": keras.layers.pooling.GlobalAveragePooling2D(
        #     pool_size, input_shape=INPUT_SHAPE
        # ),
        # "GlobalMaxPooling2D": keras.layers.pooling.GlobalMaxPooling2D(
        #     pool_size, input_shape=INPUT_SHAPE
        # ),
    }

    for layer_name, layer in layers_2D.items():
        run_rule(layer, layer_name, xs)


if __name__ == "__main__":
    generate()
