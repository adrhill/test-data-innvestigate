"""
Run an analysis of one picture on VGG16 through all analyzers.
"""
import os
from pathlib import Path

import h5py
import keras
import numpy as np
import tensorflow as tf

import innvestigate.utils.keras.graph as igraph
from innvestigate.applications.imagenet import vgg16
from test_data_innvestigate.utils.analyzers import METHODS

np.random.seed(123)
ROOT_DIR = os.path.abspath(os.curdir)
IMG_NAME = "ILSVRC2012_val_00011670.JPEG"

INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 1


def generate():
    """Load an image and run it through VGG16, applying all iNNvestigate analyzers."""
    print("Generating data on VGG16...")
    x = np.random.rand(BATCH_SIZE, *INPUT_SHAPE)

    # Create data folder
    Path(os.path.join(ROOT_DIR, "data", "vgg16")).mkdir(parents=True, exist_ok=True)

    for analyzer_name, m in METHODS.items():
        print("\t... using {}".format(analyzer_name))
        method, kwargs = m

        keras.backend.clear_session()
        session = tf.Session(graph=tf.get_default_graph())
        keras.backend.set_session(session)
        tf.set_random_seed(123)

        # Load model from innvestigate.applications
        net = vgg16(load_weights=True, load_patterns="relu")
        model = keras.models.Model(inputs=net["in"], outputs=net["sm_out"])
        model = igraph.model_wo_softmax(model)  # Strip softmax layer

        y = model.predict(x)

        # Write to hdf5 file
        data_path = os.path.join(ROOT_DIR, "data", "vgg16", analyzer_name + ".hdf5")
        with h5py.File(data_path, "w") as f:

            f.create_dataset("input", data=x)
            f.create_dataset("output", data=y)
            f.attrs["analyzer_name"] = analyzer_name
            f.attrs["model_name"] = "vgg16"
            f.attrs["input_name"] = IMG_NAME

            # Get analyzer class & construct analyzer
            analyzer = method(model, **kwargs)
            a = analyzer.analyze(x)
            f.create_dataset("attribution", data=a)


if __name__ == "__main__":
    generate()
