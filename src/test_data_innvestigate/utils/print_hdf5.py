import os

import h5py

ROOT_DIR = os.path.abspath(os.curdir)


def print_node(name, node):
    print("\t{} {}".format(name, node))


def print_hdf5(path):
    with h5py.File(path, "r") as f:
        print("Attributes:")
        [print("\t{}".format(attribute)) for attribute in f.attrs]

        print("Nodes & data:")
        f.visititems(print_node)


if __name__ == "__main__":

    path_vgg = os.path.join(ROOT_DIR, "data", "vgg16", "LRPSequentialPresetA.hdf5")
    print_hdf5(path_vgg)

    path_layer = os.path.join(ROOT_DIR, "data", "layer", "Dense.hdf5")
    print_hdf5(path_layer)
