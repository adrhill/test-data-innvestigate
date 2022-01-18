"""Run all scripts that generate test data."""
from .generate import single_layer


def main():
    """Generate all test data."""

    single_layer.generate()
