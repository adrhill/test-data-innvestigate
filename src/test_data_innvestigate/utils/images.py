import matplotlib.pyplot as plt
import numpy as np
import PIL.Image


def load_image(path, size):
    ret = PIL.Image.open(path)
    ret = ret.resize((size, size))
    ret = np.asarray(ret, dtype=np.uint8).astype(np.float32)
    if ret.ndim == 2:
        # Convert gray scale image to color channels.
        ret.resize((size, size, 1))
        ret = np.repeat(ret, 3, axis=-1)
    return ret


def show_image(image):
    fig, ax = plt.subplots(figsize=plt.figaspect(image))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.imshow(image, aspect="equal")
    return fig, ax


def show_heatmap(image):
    fig, ax = plt.subplots(figsize=plt.figaspect(image))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.imshow(image, cmap="seismic", clim=(-1, 1), aspect="equal")
    return fig, ax


def aggeregate_attribution(a):
    # Aggregate along color channels
    a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
    # Normalize to [-1, 1]
    a /= np.max(np.abs(a))
    return a