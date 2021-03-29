# test-data-innvestigate
This repo provides test data for development of the [iNNvestigate library](https://github.com/albermax/innvestigate) and future LRP implementations. [iNNvestigate release `1.0.9`](https://github.com/albermax/innvestigate/commit/b1084b2b5c59434060c78bb163b9bf006f5bbeb8) is used to generate the data.



## Installation 
To ensure reproducible generation of the data, install this repo with [Poetry](https://python-poetry.org/). Make sure you have a local installation of Python 3.6 and run:
```bash
poetry install 
```

## Usage
Generate data by running:
```bash
poetry run test-data-innvestigate
```

The generated data can then be found in the created `data` directory and is split in two folders:
* `data/vgg16`: sample input, layer-wise relevances and attributions for most analyzers and presets implemented in iNNvestigate on VGG16
* `data/layer`: **⚠️WIP⚠️** attributions from randomly generated layers

All data is saved in [HDF5 format](https://portal.hdfgroup.org/display/knowledge/What+is+HDF5). To use this data in Python, refer to the [`h5py` docs](https://docs.h5py.org/en/latest/index.html). 
<!-- **New LRP implementations should match the test datasets within a pixel-wise tolerance of `1e-5`!** -->

## File structure

### `data/vgg16`
In case the analyzer used to generate the data is of iNNvestigate's type `ReverseAnalyzerBase`, all intermediate Tensors are saved by their Node ID: 
```
├── input
├── attribution
└── layerwise_relevances/
    └── [NODE ID]
```
**Additional attributes / metadata:**
* `analyzer_name`: name of the iNNvestigate analyzer used to generate the file
* `model_name`: name of the model, in this case `vgg16`
* `input_name`: name of the input image used to generate the data

### `data/layer` 
**⚠️ WIP ⚠️**
All matching relevance rules are run on the layer. For each rule, the dataset  contains the attributions calculated from the actual output and from uniform output:
```
├── input
├── output
└── attributions/
    └── [RRule]
        ├── output
        └── uniform
```
**Additional attributes / metadata:**
* `layer_name`: name of the layer used to generate the data.
