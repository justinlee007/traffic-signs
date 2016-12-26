import numpy as np
from sklearn.preprocessing import LabelBinarizer


def one_hot_encode(input_train_labels, input_test_labels):
    # Turn labels into numbers and apply One-Hot Encoding
    encoder = LabelBinarizer()
    encoder.fit(input_train_labels)
    output_train_labels = encoder.transform(input_train_labels)
    output_test_labels = encoder.transform(input_test_labels)

    # Change to float32, so it can be multiplied against the features in TensorFlow, which are float32
    output_train_labels = output_train_labels.astype(np.float32)
    output_test_labels = output_test_labels.astype(np.float32)
    return output_train_labels, output_test_labels
