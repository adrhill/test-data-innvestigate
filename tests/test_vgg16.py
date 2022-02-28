"""
Test all LRP Analyzers over single layer models using random weights and random input.
"""
import os

import h5py
import keras
import numpy as np
import pytest
import tensorflow as tf

import innvestigate.utils.keras.graph as igraph
from innvestigate.applications.imagenet import vgg16
from test_data_innvestigate.utils.analyzers import METHODS

atol = 1e-5
rtol = 1e-3

# Loosen tolerances for SmoothGrad because of random Gaussian noise
atol_smoothgrad = 0.15
rtol_smoothgrad = 0.25

# Sizes used for data generation
input_shape = (10, 10, 3)
batch_size = 1
kernel_size = (3, 3)
pool_size = (2, 2)


def debug_failed_all_close(
    val, ref, val_name, layer_name, analyzer_name, rtol=rtol, atol=atol
):
    diff = np.absolute(val - ref)
    # Function evaluated by np.allclose, see "Notes":
    # https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
    tol = atol + rtol * np.absolute(ref)
    idx = np.argwhere(diff > tol)

    print(
        f"{len(idx)}/{np.prod(val.shape)} "
        f'failed on referece "{val_name}" using layer {layer_name} with {analyzer_name}'
        f"(atol={atol}, rtol={rtol})"
    )
    for i in idx:
        ti = tuple(i)
        print(
            f"{ti}: diff {diff[ti]} > tol {tol[ti]}"
            f"\tfor values {val_name}={val[ti]}, {val_name}_ref={ref[ti]}"
        )


@pytest.mark.vgg16
@pytest.mark.parametrize(
    "analyzer_name, val", METHODS.items(), ids=list(METHODS.keys())
)
def test_vgg16(val, analyzer_name):
    method, kwargs = val
    data_path = os.path.join(
        os.path.abspath(os.curdir),
        "data",
        "vgg16",
        analyzer_name + ".hdf5",
    )

    keras.backend.clear_session()
    # session = tf.Session(graph=tf.get_default_graph())
    # keras.backend.set_session(session)
    # tf.set_random_seed(123)

    # Load model from innvestigate.applications
    net = vgg16(load_weights=True, load_patterns="relu")
    model = keras.models.Model(inputs=net["in"], outputs=net["sm_out"])
    model = igraph.model_wo_softmax(model)  # Strip softmax layer

    with h5py.File(data_path, "r") as f:
        assert f.attrs["analyzer_name"] == analyzer_name  # sanity check: correct file
        # Model outputs should match
        x = f["input"][:]
        y = model.predict(x)

        y_ref = f["output"][:]
        outputs_match = np.allclose(y, y_ref, rtol=rtol, atol=atol)
        if not outputs_match:
            debug_failed_all_close(y, y_ref, "y", layer_name, analyzer_name)
        assert outputs_match

        # Analyze model
        analyzer = method(model, **kwargs)
        a = analyzer.analyze(x)
        assert np.shape(a) == np.shape(x)

        # Test attribution
        a_ref = f["attribution"][:]

        if analyzer_name == "SmoothGrad":
            _atol = atol_smoothgrad
            _rtol = rtol_smoothgrad
        else:
            _atol = atol
            _rtol = rtol

        attributions_match = np.allclose(a, a_ref, rtol=_rtol, atol=_atol)
        if not attributions_match:
            debug_failed_all_close(
                a, a_ref, "a", layer_name, analyzer_name, rtol=_rtol, atol=_atol
            )
        assert attributions_match
