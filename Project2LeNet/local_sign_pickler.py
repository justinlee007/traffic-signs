import os
import pickle

import matplotlib.image

import data_reader


def pickle_local_images(image_dir="./TrafficSignImages/", pickle_file="../local_test.p"):
    """
    Tool for pickling BMP files in the given image_dir.  Images must be 32x32x3 and have a classifier at the end.
    :param image_dir: must contain only BMP files
    :param pickle_file: output pickle
    :return:
    """
    data = {"features": [], "labels": []}
    file = open(pickle_file, "wb")
    for filename in os.listdir(image_dir):
        image = matplotlib.image.imread(image_dir + filename, "bmp")
        label = filename[(filename.find('-') + 1):filename.find('.')]
        print("Appending {} with label={}".format(filename, label))
        data["features"].append(image)
        data["labels"].append(label)
    pickle.dump(data, file)
    file.close()


if __name__ == '__main__':
    pickle_local_images()
    data = data_reader.unpickle_files("../train.p", "../local_test.p")
