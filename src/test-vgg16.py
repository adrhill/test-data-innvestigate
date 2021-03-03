import os
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image



import innvestigate
import innvestigate.utils
import keras.applications.vgg16 as vgg16


if __name__ == "__main__":
    # Load an image.
    # Need to download examples images first.
    # See script in images directory.

    base_dir = os.path.dirname(__file__)
    image = load_image(
        os.path.join(base_dir, "images", "ILSVRC2012_val_00011670.JPEG"), 224)

    # Code snippet.
    fig, ax = show_image(image/255)
    plt.show()
    plt.savefig("readme_example_input.png")

    # Get model
    model, preprocess = vgg16.VGG16(), vgg16.preprocess_input
    # Strip softmax layer
    model = innvestigate.utils.model_wo_softmax(model)

    # Create analyzer
    analyzer = innvestigate.create_analyzer("deep_taylor", model)

    # Add batch axis and preprocess
    x = preprocess(image[None])
    # Apply analyzer w.r.t. maximum activated output-neuron
    a = analyzer.analyze(x)

    # Aggregate along color channels and normalize to [-1, 1]
    a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
    a /= np.max(np.abs(a))

    # Plot
    fig, ax = show_heatmap(a[0])
    plt.show()
    plt.savefig("readme_example_analysis.png")