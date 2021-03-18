import os

from .generate import vgg16, single_layer

ROOT_DIR = os.path.abspath(os.curdir)


def main():
    """Generate all test data."""

    vgg16.generate()
    single_layer.generate()
