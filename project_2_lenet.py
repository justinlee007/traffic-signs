"""
LeNet Architecture for GTSRB dataset
"""
import sys

import tensorflow as tf
from tensorflow.contrib.layers import flatten
from tqdm import tqdm

import data_reader

NUM_EPOCHS = 100
BATCH_SIZE = 64


# LeNet architecture:
# INPUT -> CONV -> ACT -> POOL -> CONV -> ACT -> POOL -> FLATTEN -> FC -> ACT -> FC
def LeNet(x, y):
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
    # Given a dataset as input returns the loss and accuracy.
    steps_per_epoch = len(features) // BATCH_SIZE
    num_examples = steps_per_epoch * BATCH_SIZE
    total_acc, total_loss = 0, 0
    for step in range(steps_per_epoch):
        batch_features, batch_labels = next_batch(step, features, labels)
        loss, acc = sess.run([loss_op, accuracy_op], feed_dict={x: batch_features, y: batch_labels})
        total_acc += (acc * len(batch_features))
        total_loss += (loss * len(batch_features))
    return total_loss / num_examples, total_acc / num_examples


def next_batch(step, features, labels):
    batch_start = step * BATCH_SIZE
    batch_features = features[batch_start:batch_start + BATCH_SIZE]
    batch_labels = labels[batch_start:batch_start + BATCH_SIZE]
    return batch_features, batch_labels


if __name__ == '__main__':
    # Load data
    mnist = data_reader.read_pickle_sets()
    train_features = mnist["train_features"]
    train_labels = mnist["train_labels"]
    valid_features = mnist["valid_features"]
    valid_labels = mnist["valid_labels"]
    test_features = mnist["test_features"]
    test_labels = mnist["test_labels"]

    num_classes = mnist["num_classes"]
    image_shape = train_features[0].shape[0]

    # Dataset consists of 32x32x3, color images.
    x = tf.placeholder(tf.float32, (None, image_shape))
    # Classify over 43 labels.
    y = tf.placeholder(tf.float32, (None, num_classes))
    # Create the LeNet.
    fc2 = LeNet(x, num_classes)

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc2, y))
    opt = tf.train.AdamOptimizer()
    train_op = opt.minimize(loss_op)
    correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(y, 1))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    steps_per_epoch = len(train_features) // BATCH_SIZE
    print("BATCH_SIZE={}, EPOCHS={}, steps_per_epoch={}, image_shape={}".format(
        BATCH_SIZE, NUM_EPOCHS, steps_per_epoch, image_shape))

    with tf.Session() as sess:
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

        # Evaluate on the test data
        test_loss, test_acc = eval_data(test_features, test_labels)
        print("Test loss={:.4f}, Test accuracy={:.4f}".format(test_loss, test_acc))
