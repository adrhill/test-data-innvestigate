import os

import innvestigate
import innvestigate.utils

import keras.applications.vgg16 as vgg16

import matplotlib.pyplot as plt

import numpy as np

from ..utils.images import load_image, show_heatmap, show_image

ROOT_DIR = os.path.abspath(os.curdir)


def generate():
    """Load an image and run it through VGG16, applying LRP."""
    # Load image
    image = load_image(
        os.path.join(
            ROOT_DIR,
            "src",
            "test_data_innvestigate",
            "assets",
            "ILSVRC2012_val_00011670.JPEG",
        ),
        size=224,
    )

    # Get model
    model, preprocess = vgg16.VGG16(), vgg16.preprocess_input
    # Strip softmax layer
    model = innvestigate.utils.model_wo_softmax(model)

    # Create analyzer
    analyzer = innvestigate.analyzer.DeepTaylor(model, reverse_keep_tensors=True)

    # Add batch axis and preprocess
    x = preprocess(image[None])
    # Apply analyzer w.r.t. maximum activated output-neuron
    a = analyzer.analyze(x)

    # Obtain layerwise tensors, stripping first
    reversed_tensors = analyzer._reversed_tensors[1:]

    # unzip reverse tensors to strip indices
    tensor_indices, reversed_tensors = zip(*reversed_tensors)

    assert np.allclose(
        reversed_tensors[0], a
    ), "_reversed_tensors output differs from attribution"

    # Save data
    data_path = os.path.join(ROOT_DIR, "data", "models", "vgg16")
    np.savez_compressed(data_path, *reversed_tensors)
