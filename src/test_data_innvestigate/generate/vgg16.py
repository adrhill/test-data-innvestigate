import os
from pathlib import Path

import h5py

# import innvestigate
# import innvestigate.analyzer.relevance_based.relevance_analyzer as ranalyzer
import innvestigate.analyzer as iAnalyzers
import innvestigate.utils as iutils
from innvestigate.analyzer import BoundedDeepTaylor, PatternNet
from innvestigate.analyzer.base import ReverseAnalyzerBase
from innvestigate.applications.imagenet import vgg16

import keras

import numpy as np

from test_data_innvestigate.utils.analyzers import ANALYZERS
from test_data_innvestigate.utils.images import load_image


ROOT_DIR = os.path.abspath(os.curdir)
IMG_NAME = "ILSVRC2012_val_00011670.JPEG"


def generate():
    """Load an image and run it through VGG16, applying LRP."""
    print("Generating data on VGG16...")

    # Create data folder
    Path(os.path.join(ROOT_DIR, "data", "vgg16")).mkdir(parents=True, exist_ok=True)

    # Load image
    image = load_image(
        os.path.join(
            ROOT_DIR,
            "src",
            "test_data_innvestigate",
            "assets",
            IMG_NAME,
        ),
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
    model = iutils.keras.graph.model_wo_softmax(model)

    # Add batch axis and preprocess input
    x = preprocess(image[None])

    for analyzer_name in ANALYZERS:
        # Write to hdf5 file
        data_path = os.path.join(ROOT_DIR, "data", "vgg16", analyzer_name + ".hdf5")
        with h5py.File(data_path, "w") as f:

            f.create_dataset("input", data=x)
            f.attrs["analyzer_name"] = analyzer_name
            f.attrs["model_name"] = "vgg16"
            f.attrs["input_name"] = IMG_NAME

            # Get analyzer class & construct analyzer
            Analyzer = getattr(iAnalyzers, analyzer_name)

            if issubclass(Analyzer, PatternNet):
                analyzer = Analyzer(model, patterns=patterns)
            elif issubclass(Analyzer, BoundedDeepTaylor):
                analyzer = Analyzer(model, low=input_range[0], high=input_range[1])
            else:
                analyzer = Analyzer(model)

            print("\t... using {}: {}".format(analyzer_name, analyzer))

            # Im method reverses model, keep track of tensors on the backward-pass
            if issubclass(Analyzer, ReverseAnalyzerBase):
                print("\t\t keeping track of backwards-pass")
                analyzer._reverse_keep_tensors = True

                # Apply analyzer w.r.t. maximum activated output-neuron
                a = analyzer.analyze(x)
                f.create_dataset("attribution", data=a)

                # Obtain layerwise tensors
                relevances = analyzer._reversed_tensors
                # unzip reverse tensors to strip indices
                indices, relevances = zip(*relevances)

                assert np.allclose(
                    relevances[1], a
                ), "_reversed_tensors output differs from final attribution"

                # Save relevances
                f_rel = f.create_group("layerwise_relevances")
                for idx, rel in zip(indices, relevances):
                    f_rel.create_dataset(str(idx[0]), data=rel)
            else:
                a = analyzer.analyze(x)
                f.create_dataset("attribution", data=a)


if __name__ == "__main__":
    generate()
