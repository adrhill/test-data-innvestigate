import os

import h5py
import pytest

from test_data_innvestigate.generate import vgg16

# Load data
ROOT_DIR = os.path.abspath(os.curdir)
path = os.path.join(ROOT_DIR, "data", "vgg16", "LRPSequentialPresetA.hdf5")
data = h5py.File(path, "r")
rels = data["layerwise_relevances"]

# Define expected shapes for comparison
shapes_vgg16 = [
    (1, 224, 224, 3),
    (1, 224, 224, 64),
    (1, 224, 224, 64),
    (1, 112, 112, 64),
    (1, 112, 112, 128),
    (1, 112, 112, 128),
    (1, 56, 56, 128),
    (1, 56, 56, 256),
    (1, 56, 56, 256),
    (1, 56, 56, 256),
    (1, 28, 28, 256),
    (1, 28, 28, 512),
    (1, 28, 28, 512),
    (1, 28, 28, 512),
    (1, 14, 14, 512),
    (1, 14, 14, 512),
    (1, 14, 14, 512),
    (1, 14, 14, 512),
    (1, 7, 7, 512),
    (1, 25088),
    (1, 4096),
    (1, 4096),
    (1, 1000),
]

# Get shape of attributions on individual layers
n_layers = len(shapes_vgg16)
shapes_data = [rels.get(str(l)).shape for l in range(n_layers)]


@pytest.mark.parametrize("shape_data, shape_expected", zip(shapes_data, shapes_vgg16))
def test_shape_match(shape_data, shape_expected):
    assert shape_data == shape_expected
