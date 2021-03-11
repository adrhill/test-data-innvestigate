import os

from .generate import vgg16

ROOT_DIR = os.path.abspath(os.curdir)


def main():
    """Generate all test data."""

    vgg16.generate()
