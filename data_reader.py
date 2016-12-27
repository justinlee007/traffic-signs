import pickle

import grayscale as gs
import one_hot_encoder as ohe
import splitter


# Unpickle the images and pre-process them
def read_pickle_sets(train_dir="./"):
    # Load pickled data
    training_file = train_dir + "train.p"
    testing_file = train_dir + "test.p"

    print("Loading training_file=\"{}\"".format(training_file))
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    print("Loading testing_file=\"{}\"".format(testing_file))
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    # Delineate features and labels
    train_features, train_labels = train['features'], train['labels']
    test_features, test_labels = test['features'], test['labels']

    # Total number of labels
    classes = set(train_labels)
    num_classes = len(classes)

    # Convert to grayscale
    train_features = gs.grayscale(train_features)
    test_features = gs.grayscale(test_features)

    # Normalize from 0-255 to 0.1-0.9
    train_features = gs.normalize_grayscale(train_features)
    test_features = gs.normalize_grayscale(test_features)

    # Reshape into height * width
    train_features = train_features.reshape(train_features.shape[0], train_features.shape[1] * train_features.shape[2])
    test_features = test_features.reshape(test_features.shape[0], test_features.shape[1] * test_features.shape[2])

    # One-hot encode labels
    train_labels, test_labels = ohe.one_hot_encode(train_labels, test_labels)

    # Randomly split up validation set
    train_features, valid_features, train_labels, valid_labels = splitter.split_data_sets(train_features, train_labels)
    num_train = len(train_features)
    num_valid = len(valid_features)
    num_test = len(test_features)

    print("num_train={}, num_valid={}, num_test={}, num_classes={}".format(num_train, num_valid, num_test, num_classes))
    return {"train_features": train_features, "train_labels": train_labels,
            "valid_features": valid_features, "valid_labels": valid_labels,
            "test_features": test_features, "test_labels": test_labels,
            "num_classes": num_classes}