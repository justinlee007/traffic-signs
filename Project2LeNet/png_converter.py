import csv

import matplotlib.image
import matplotlib.pyplot as plt

def convert_to_array():
    signnames = []
    with open("signnames.csv", "rt") as csvfile:
        signreader = csv.reader(csvfile, delimiter=',')
        for index, row in enumerate(signreader):
            if index == 0:
                continue
            signnames.append(row[1])

    print(signnames)
    image = matplotlib.image.imread("./TrafficSignImages/20161128_164531.bmp", "bmp")
    print("image={}".format(image))

convert_to_array()
