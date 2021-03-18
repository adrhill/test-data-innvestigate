import os

import h5py

import innvestigate
import innvestigate.analyzer.relevance_based.relevance_rule as rrule
import innvestigate.utils.keras.checks as iutils
from innvestigate.analyzer.relevance_based.relevance_analyzer import LRP

import keras
import keras.backend as K

import matplotlib.pyplot as plt

import numpy as np

from ..utils.images import show_heatmap, show_image

ROOT_DIR = os.path.abspath(os.curdir)
LRP_RULES = rrule.__all__

"""
Create input data representative of a single 2D image with 3 channels
Using the default `data_format="channels_last"`: 
    (batch_size, rows, cols, channels)
"""
INPUT_SHAPE = (10, 10, 3)
BATCH_SIZE = 1


def get_single_layer_model(layer):
    x = keras.layers.Input(shape=INPUT_SHAPE)
    y = layer(x)

    model = keras.Model(x, y)
    weights = model.layers[0].get_weights()

    return model, weights


def run_rule(layer, layer_name: str, xs: np.ndarray):
    print("Generating data on layer {}".format(layer_name))
    # K.clear_session()

    # Get model
    model, weights = get_single_layer_model(layer)
    print(model.input_shape)

    # Get prediction
    ys = model.predict(xs, steps=BATCH_SIZE)
    uniform = np.ones_like(ys)

    print(np.shape(ys))
    print(type(ys))

    # Create hdf5 data
    data_path = os.path.join(ROOT_DIR, "data", "layer", layer_name + ".hdf5")

    with h5py.File(data_path, "w") as f:
        f.attrs["layer_name"] = layer_name
        f.create_dataset("input", data=xs)
        f.create_dataset("output", data=ys)
        rels = f.create_group("relevances")

        # Apply all rules on ys and uniform output
        for rule_name in LRP_RULES:
            # state and reverse_state are required to construct and apply LRP-rules respectively
            # TODO: check if these can be left to `None`
            state = None
            reverse_state = None

            # Create ReverseAnalyzerBase using rule
            Rule = getattr(rrule, rule_name)  # get class of rule
            rule = Rule(layer, state)
            print("\t... using {}: {}".format(rule_name, rule))
            analyzer = LRP(model, rule=[rule], input_layer_rule=rule)

            # Analyze output
            a = analyzer.analyze(xs)

            # Get rule matching name from iNNvestigate relevance_rules
            r = rels.create_group(rule_name)

            # Calculate relevances for output Rs=ys of layers
            rel_out = rule.apply(xs, ys, ys, reverse_state)
            r.create_dataset("output", data=rel_out)

            # Calculate relevances for uniform Rs
            rel_unif = rule.apply(xs, ys, uniform, reverse_state)  # TODO: keep xs?
            r.create_dataset("uniform", data=rel_unif)


def generate():
    """Create single layer model and run LRP"""

    xs = np.random.rand(BATCH_SIZE, *INPUT_SHAPE)

    # Create layers that are evaluated
    kernel_size = (3, 3)
    pool_size = (2, 2)

    layers_2D = {
        "Dense": keras.layers.Dense(10, input_shape=INPUT_SHAPE),
        "Dense_relu": keras.layers.Dense(
            10, activation="relu", input_shape=INPUT_SHAPE
        ),
        "Conv2D": keras.layers.convolutional.Conv2D(
            10, kernel_size, input_shape=INPUT_SHAPE
        ),
        "Conv2D_relu": keras.layers.convolutional.Conv2D(
            10, kernel_size, activation="relu", input_shape=INPUT_SHAPE
        ),
        "AveragePooling2D": keras.layers.pooling.AveragePooling2D(
            pool_size, input_shape=INPUT_SHAPE
        ),
        "MaxPooling2D": keras.layers.pooling.MaxPooling2D(
            pool_size, input_shape=INPUT_SHAPE
        ),
        # "GlobalAveragePooling2D": keras.layers.pooling.GlobalAveragePooling2D(
        #     pool_size, input_shape=INPUT_SHAPE
        # ),
        # "GlobalMaxPooling2D": keras.layers.pooling.GlobalMaxPooling2D(
        #     pool_size, input_shape=INPUT_SHAPE
        # ),
    }

    for layer_name, layer in layers_2D.items():
        run_rule(layer, layer_name, xs)
