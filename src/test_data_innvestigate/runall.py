import os

from .generate import vgg16
from .utils.images import load_image

BASE_DIR = os.path.dirname(__file__)


def main():
    """
    Load sample image and run it through models.
    """

    image = load_image(
        os.path.join(BASE_DIR, "assets", "ILSVRC2012_val_00011670.JPEG"), size=224
    )

    vgg16.run(image, show_plot=True)
