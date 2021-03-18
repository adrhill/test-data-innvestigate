import os

import innvestigate
import innvestigate.analyzer.relevance_based.relevance_analyzer as ranalyzer

import h5py

import keras.applications.vgg16 as vgg16

import matplotlib.pyplot as plt

import numpy as np

from ..utils.images import load_image, show_heatmap, show_image

ROOT_DIR = os.path.abspath(os.curdir)
LRP_ANALYZERS = [
    "LRPZ",
    "LRPZIgnoreBias",
    "LRPEpsilon",
    "LRPEpsilonIgnoreBias",
    "LRPWSquare",
    "LRPFlat",
    "LRPAlpha2Beta1",
    "LRPAlpha2Beta1IgnoreBias",
    "LRPAlpha1Beta0",
    "LRPAlpha1Beta0IgnoreBias",
    "LRPZPlus",
    "LRPZPlusFast",
    "LRPSequentialPresetA",
    "LRPSequentialPresetB",
    "LRPSequentialPresetAFlat",
    "LRPSequentialPresetBFlat",
    "LRPSequentialPresetBFlatUntilIdx",
]


def generate():
    """Load an image and run it through VGG16, applying LRP."""
    print("Generating data on VGG16...")

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

    # Add batch axis and preprocess
    x = preprocess(image[None])

    # Write to hdf5 file
    data_path = os.path.join(ROOT_DIR, "data", "models", "vgg16.hdf5")
    with h5py.File(data_path, "w") as f:
        f.create_dataset("input", data=x)
        f.attrs["model_name"] = "vgg16"

        rels = f.create_group("layerwise_relevances")

        # Run all analyzers
        for analyzer_name in LRP_ANALYZERS:
            # Get analyzer class
            Analyzer = getattr(ranalyzer, analyzer_name)

            # Construct analyzer
            analyzer = Analyzer(model, reverse_keep_tensors=True)
            print("\t... using {}: {}".format(analyzer_name, analyzer))

            # Apply analyzer w.r.t. maximum activated output-neuron
            a = analyzer.analyze(x)

            # Obtain layerwise tensors
            relevances = analyzer._reversed_tensors
            # unzip reverse tensors to strip indices
            indices, relevances = zip(*relevances)

            assert np.allclose(
                relevances[1], a
            ), "_reversed_tensors output differs from final attribution"

            # Save relevances for this analyzer
            ana = rels.create_group(analyzer_name)
            for idx, rel in zip(indices, relevances):
                ana.create_dataset(str(idx[0]), data=rel)
