# test-data-innvestigate
The aim of this repo is to provide test data for future LRP implementations. For this purpose, the [iNNvestigate library](https://github.com/albermax/innvestigate) release `1.0.9` is used as a "ground truth implementation".

## Usage
The generated data can be found in `/data` and is split in two folders:
* `data/unit-input`: relevances of randomly generated layers for unit-inputs
* `data/models`: sample inputs, activations and relevances for common models, e.g.:
    * VGG16


**New LRP implementations should match the test datasets within a pixel-wise tolerance of `1e-5`!**

## Installation 
**This package doesn't need to be installed, as the generated data are already part of the repo.**

However, to allow reproducible generation of the data, you can install the package with [Poetry](https://python-poetry.org/). Make sure you have a local installation of Python 3.6 (e.g. with pyenv) and run:
```bash
poetry install 
poetry run test-data-innvestigate
```

### Why these specific package versions?
The versions are set according to the [iNNvestigate installation instructions](https://github.com/albermax/innvestigate#installation) for [version `1.0.9`](https://github.com/albermax/innvestigate/commit/b1084b2b5c59434060c78bb163b9bf006f5bbeb8), which is tested using:
* Python `3.6`
* TensorFlow `1.12`
* CUDA `9.x`

Additionally:
* using NumPy `1.16` to avoid [warnings with Tensorflow `1.12`](https://github.com/tensorflow/tensorflow/issues/31249) 