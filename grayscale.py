import cv2
import numpy as np


def grayscale(input_features):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    output_features = []
    for image in input_features:
        output_features.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    return np.array(output_features)


# Min-Max scaling for grayscale image data
def normalize_grayscale(input_features):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data

    """
    output_features = []
    for image in input_features:
        min_val = 0
        max_val = 255
        a = 0.1
        b = 0.9
        output_features.append(a + (((image - min_val) * (b - a)) / (max_val - min_val)))
    return np.array(output_features)


"""
def flatten_features(input_features):
    output_features = []
    for image in input_features:
        output_features.append(np.array(image, dtype=np.float32).flatten())
    return output_features
"""
