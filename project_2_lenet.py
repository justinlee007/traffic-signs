"""
LeNet Architecture for GTSRB dataset
"""

import argparse
import sys
import time

import tensorflow as tf
from tensorflow.contrib.layers import flatten
from tqdm import tqdm

import data_reader

NUM_EPOCHS = 2
BATCH_SIZE = 64
UPPER_THRESHOLD = 0.988


def create_lenet(x, y):
    """
    LeNet architecture: INPUT -> CONV -> ACT -> POOL -> CONV -> ACT -> POOL -> FLATTEN -> FC -> ACT -> FC
    :param x:
    :param y:
    :return:
    """
    # Reshape from 2D to 4D. This prepares the data for convolutional and pooling layers.
    x = tf.reshape(x, (-1, 32, 32, 1))
    # Squish values from 0-255 to 0-1.
    x /= 255.
    # Resize to 32x32.
    x = tf.image.resize_images(x, (32, 32))

    # 28x28x6
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6)))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    conv1 = tf.nn.relu(conv1)

    # 14x14x6
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # 10x10x16
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16)))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    conv2 = tf.nn.relu(conv2)

    # 5x5x16
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten
    fc1 = flatten(conv2)
    # (5 * 5 * 16, 120)
    fc1_shape = (fc1.get_shape().as_list()[-1], 120)

    fc1_W = tf.Variable(tf.truncated_normal(shape=(fc1_shape)))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc1, fc1_W) + fc1_b
    fc1 = tf.nn.relu(fc1)

    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, y)))
    fc2_b = tf.Variable(tf.zeros(y))
    return tf.matmul(fc1, fc2_W) + fc2_b


def eval_data(features, labels):
    """
    Given a dataset as input returns the loss and accuracy.
    :param features:
    :param labels:
    :return:
    """
    total_accuracy, total_loss = 0, 0
    if len(features) < BATCH_SIZE:
        total_loss, total_accuracy = sess.run([loss_op, accuracy_op], feed_dict={x: features, y: labels})
    else:
        steps_per_epoch = len(features) // BATCH_SIZE
        num_examples = steps_per_epoch * BATCH_SIZE
        for step in range(steps_per_epoch):
            batch_features, batch_labels = next_batch(step, features, labels)
            loss, acc = sess.run([loss_op, accuracy_op], feed_dict={x: batch_features, y: batch_labels})
            total_accuracy += (acc * len(batch_features))
            total_loss += (loss * len(batch_features))
        total_loss /= num_examples
        total_accuracy /= num_examples
    return total_loss, total_accuracy


def next_batch(step, features, labels):
    batch_start = step * BATCH_SIZE
    batch_features = features[batch_start:batch_start + BATCH_SIZE]
    batch_labels = labels[batch_start:batch_start + BATCH_SIZE]
    return batch_features, batch_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LeNet Architecture for GTSRB dataset")
    parser.add_argument("-train", action="store", dest="train_file", default="train.p", help="Training Pickle")
    parser.add_argument("-test", action="store", dest="test_file", default="test.p", help="Testing pickle")
    parser.add_argument("-s", action="store", dest="save_file", help="Tensor save file to restore")
    results = parser.parse_args()
    train_file = results.train_file
    test_file = results.test_file
    save_file = results.save_file

    # Load data
    data = data_reader.read_pickle_sets(train_file, test_file)
    train_features = data["train_features"]
    train_labels = data["train_labels"]
    valid_features = data["valid_features"]
    valid_labels = data["valid_labels"]
    test_features = data["test_features"]
    test_labels = data["test_labels"]
    num_classes = data["num_classes"]
    image_shape = train_features[0].shape[0]

    # Dataset consists of 32x32x3, color images.
    x = tf.placeholder(tf.float32, (None, image_shape))
    # Classify over 43 labels.
    y = tf.placeholder(tf.float32, (None, num_classes))
    # Create the LeNet.
    fc2 = create_lenet(x, num_classes)

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc2, y))
    opt = tf.train.AdamOptimizer()
    train_op = opt.minimize(loss_op)
    correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(y, 1))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    steps_per_epoch = len(train_features) // BATCH_SIZE
    print("BATCH_SIZE={}, EPOCHS={}, steps_per_epoch={}, image_shape={}".format(
        BATCH_SIZE, NUM_EPOCHS, steps_per_epoch, image_shape))

    # Class used to save and/or restore Tensor Variables
    saver = tf.train.Saver()

    with tf.Session() as sess:
        if save_file is None:
            sess.run(tf.initialize_all_variables())

            # Progress bar
            epochs = tqdm(range(NUM_EPOCHS), desc="Training Model", file=sys.stdout, unit="Epoch")

            # Train model
            for i in epochs:
                # Loop over all batches
                for step in range(steps_per_epoch):
                    batch_features, batch_labels = next_batch(step, train_features, train_labels)
                    loss = sess.run(train_op, feed_dict={x: batch_features, y: batch_labels})

                val_loss, val_acc = eval_data(valid_features, valid_labels)
                epochs.write(
                    "Epoch {}: Validation loss={:.4f}, Validation accuracy={:.4f}".format((i + 1), val_loss, val_acc))
                if val_acc > UPPER_THRESHOLD:
                    epochs.write("Threshold reached: {}".format(UPPER_THRESHOLD))
                    epochs.close()
                    break

            # Save the model
            save_file = "train_model-" + time.strftime("%Y%m%d-%H%M%S" + ".ckpt")
            saver.save(sess, save_file)
            epochs.write("Trained Model Saved.")

            # Evaluate on the test data
            test_loss, test_acc = eval_data(test_features, test_labels)
            epochs.write("Test loss={:.4f}, Test accuracy={:.4f}".format(test_loss, test_acc))
        else:
            print("Restoring session from {}".format(save_file))
            saver.restore(sess, save_file)

            # Evaluate on the test data
            test_loss, test_acc = eval_data(test_features, test_labels)
            print("Test loss={:.4f}, Test accuracy={:.4f}".format(test_loss, test_acc))
