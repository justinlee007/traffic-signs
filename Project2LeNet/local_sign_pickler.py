import os
import pickle

import matplotlib.image

import data_reader


def pickle_local_images(image_dir="./TrafficSignImages/", pickle_file="../local_test.p"):
    data = {"features": [], "labels": []}
    file = open(pickle_file, "wb")
    for filename in os.listdir(image_dir):
        image = matplotlib.image.imread(image_dir + filename, "bmp")
        data["features"].append(image)
        label = filename[(filename.find('-') + 1):filename.find('.')]
        # print("filename={}, label={}".format(filename, label))
        data["labels"].append(label)
    print(data)
    pickle.dump(data, file)
    file.close()


if __name__ == '__main__':
    pickle_local_images()
    data = data_reader.unpickle_files("../train.p", "../local_test.p")
