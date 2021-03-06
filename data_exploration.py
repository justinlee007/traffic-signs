import csv

import matplotlib.pyplot as plt

import data_reader
import project_2_lenet


def read_signnames(filename="signnames.csv"):
    # Read signnames.csv and create a list of labels
    signnames = []
    with open(filename, "rt") as csvfile:
        signreader = csv.reader(csvfile, delimiter=',')
        for index, row in enumerate(signreader):
            if index == 0:
                continue
            signnames.append(row[1])
    return signnames


def plot_signs():
    data = data_reader.unpickle_files()
    x_train = data["train_features"]
    y_train = data["train_labels"]
    signnames = read_signnames()

    # Show an example of each class of sign with its corresponding label
    current_index = 0
    current_type = 0
    num_columns = 5
    num_rows = 9

    fig, axarr = plt.subplots(num_rows, num_columns, figsize=(15, 15))
    fig.suptitle("Example Images For Each Sign Type", fontsize=14)

    for row in range(num_rows):
        for column in range(num_columns):
            image = axarr[row, column]
            image.axis('off')
            for index in range(current_index, len(y_train)):
                if y_train[index] == current_type:
                    current_index = index
                    image.imshow(x_train[current_index])
                    image.set_title(signnames[current_type], fontsize=8)
                    current_type += 1
                    break

    plt.show()


def plot_local_signs():
    data = data_reader.unpickle_files(training_file="local_test.p")
    x_train = data["train_features"]
    y_train = data["train_labels"]
    signnames = read_signnames()

    # Show an example of each class of sign with its corresponding label
    current_index = 0
    num_columns = 5
    num_rows = 3

    fig, axarr = plt.subplots(num_rows, num_columns, figsize=(15, 15))
    fig.suptitle("Example Images For Each Sign Type", fontsize=14)

    for row in range(num_rows):
        for column in range(num_columns):
            image = axarr[row, column]
            image.axis('off')
            image.imshow(x_train[current_index])
            image.set_title(signnames[int(y_train[current_index])], fontsize=8)
            current_index += 1
    plt.show()


def plot_local_probs():
    top_prob = project_2_lenet.run_lenet(test_file="local_test.p", save_file="./train_model-20161226-051204.ckpt")
    prob_values = top_prob.values
    prob_labels = top_prob.indices

    data = data_reader.unpickle_files(training_file="local_test.p")
    x_train = data["train_features"]
    y_train = data["train_labels"]
    signnames = read_signnames()

    # Show an example of each class of sign with its corresponding label
    current_index = 0
    num_columns = 5
    num_rows = 3

    fig, axarr = plt.subplots(num_rows, num_columns, figsize=(15, 15))
    fig.suptitle("Example Images For Each Sign Type", fontsize=14)

    for row in range(num_rows):
        for column in range(num_columns):
            image = axarr[row, column]
            image.axis('off')
            image.imshow(x_train[current_index])
            current_value = prob_values[current_index]
            current_label = prob_labels[current_index]
            label = signnames[int(y_train[current_index])]
            full_label = "{} (actual)\n{} ({:.1%})\n{} ({:.1%})\n{} ({:.1%})".format(
                label,
                signnames[current_label[0]], current_value[0],
                signnames[current_label[1]], current_value[1],
                signnames[current_label[2]], current_value[2])
            image.set_title(full_label, fontsize=8)
            current_index += 1
    plt.show()


if __name__ == '__main__':
    # plot_signs()
    # plot_local_signs()
    plot_local_probs()
