"""Run all scripts that generate test data."""
from .generate import single_layer
from .generate import vgg16


def main():
    """Generate all test data."""

    single_layer.generate()
    vgg16.generate()
