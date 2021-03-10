import os

import innvestigate
import innvestigate.utils

import keras.applications.vgg16 as vgg16

import matplotlib.pyplot as plt

import numpy as np

from ..utils.images import show_heatmap, show_image

base_dir = os.path.dirname(__file__)

print(base_dir)


def run(image, show_plot=False):
    """Load an image and run it through VGG16, applying LRP."""

    # Code snippet.
    fig, ax = show_image(image / 255)
    if show_plot is True:
        plt.show()

    # Get model
    model, preprocess = vgg16.VGG16(), vgg16.preprocess_input
    # Strip softmax layer
    model = innvestigate.utils.model_wo_softmax(model)

    # Create analyzer
    analyzer = innvestigate.create_analyzer("deep_taylor", model)

    # Add batch axis and preprocess
    x = preprocess(image[None])
    # Apply analyzer w.r.t. maximum activated output-neuron
    a = analyzer.analyze(x)

    # Aggregate along color channels and normalize to [-1, 1]
    a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
    a /= np.max(np.abs(a))

    # Plot
    fig, ax = show_heatmap(a[0])
    if show_plot is True:
        plt.show()
