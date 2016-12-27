import csv

import matplotlib.image


def convert_to_array():
    signnames = []
    with open("signnames.csv", "rt") as csvfile:
        signreader = csv.reader(csvfile, delimiter=',')
        for index, row in enumerate(signreader):
            if index == 0:
                continue
            signnames.append(row[1])

    print(signnames)
    image = matplotlib.image.imread("./TrafficSignImages/20161128_164531-14.bmp", "bmp")
    print("image={}".format(image))


if __name__ == '__main__':
    convert_to_array()
