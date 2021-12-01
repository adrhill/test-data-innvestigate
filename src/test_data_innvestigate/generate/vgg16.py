"""
Run an analysis of one picture on VGG16 through all analyzers.
"""
import os
from pathlib import Path

import h5py
import keras

import innvestigate.utils.keras.graph as igraph
from innvestigate.applications.imagenet import vgg16
from test_data_innvestigate.utils.analyzers import METHODS
from test_data_innvestigate.utils.images import load_image

ROOT_DIR = os.path.abspath(os.curdir)
IMG_NAME = "ILSVRC2012_val_00011670.JPEG"


def generate():
    """Load an image and run it through VGG16, applying all iNNvestigate analyzers."""
    print("Generating data on VGG16...")

    # Create data folder
    Path(os.path.join(ROOT_DIR, "data", "vgg16")).mkdir(parents=True, exist_ok=True)

    # Load image
    image = load_image(
        os.path.join(ROOT_DIR, "src", "test_data_innvestigate", "assets", IMG_NAME),
        size=224,
    )

    # Get model with patterns and preprocessing function
    net = vgg16(load_weights=True, load_patterns="relu")
    model = keras.models.Model(inputs=net["in"], output=net["sm_out"])
    model.compile(optimizer="adam", loss="categorical_crossentropy")

    patterns = net["patterns"]
    preprocess = net["preprocess_f"]
    input_range = net["input_range"]

    # Strip softmax layer
    model = igraph.model_wo_softmax(model)

    # Add batch axis and preprocess input
    x = preprocess(image[None])

    for analyzer_name, m in METHODS.items():
        print("\t... using {}".format(analyzer_name))
        method, kwargs = m

        # Write to hdf5 file
        data_path = os.path.join(ROOT_DIR, "data", "vgg16", analyzer_name + ".hdf5")
        with h5py.File(data_path, "w") as f:

            f.create_dataset("input", data=x)
            f.attrs["analyzer_name"] = analyzer_name
            f.attrs["model_name"] = "vgg16"
            f.attrs["input_name"] = IMG_NAME

            # Get analyzer class & construct analyzer
            analyzer = method(model, **kwargs)
            a = analyzer.analyze(x)
            f.create_dataset("attribution", data=a)


if __name__ == "__main__":
    generate()
