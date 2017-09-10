"""
Udacity Deep Learning course by Google.
Assignment #03: various regularization techniques.

"""

import os
import sys
import logging
import argparse
import numpy as np
import tensorflow as tf

from utils import *
from dnn_utils import *
from dataset_utils import *


logger = logging.getLogger(os.path.basename(__file__))  # pylint: disable=invalid-name


def parse_args():
    """Parse command line args using argparse."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sanitized',
        action='store_true',
        help='Use sanitized version of the test and validation datasets',
    )
    return parser.parse_args()

def train_logistic_classifier(train_dataset, train_labels, test_dataset, test_labels, reg_coeff):
    """Standard logistic classifier with no nonlinearity."""
    batch_size = 128

    graph = tf.Graph()
    with graph.as_default():
        tf_train_dataset = tf.placeholder(tf.float32, shape=(None, IMAGE_RES * IMAGE_RES))
        tf_train_labels = tf.placeholder(tf.float32, shape=(None, NUM_CLASSES))
        tf_test_dataset = tf.constant(test_dataset)

        weights = tf.Variable(tf.truncated_normal([IMAGE_RES * IMAGE_RES, NUM_CLASSES]))
        biases = tf.Variable(tf.zeros([NUM_CLASSES]))
        logits = tf.matmul(tf_train_dataset, weights) + biases
        unregularized_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits),
        )
        l2_loss = reg_coeff * l2(weights)
        loss = unregularized_loss + l2_loss

        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
        train_prediction = tf.nn.softmax(logits)
        test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        logger.info('Initialized')
        for step in range(10001):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]

            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, predictions = sess.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

            verbose = False
            if step % 500 == 0:
                if verbose:
                    logger.info('Batch loss at step %d is %f', step, l)
                    logger.info('Batch accuracy: %.1f%%', calc_accuracy(predictions, batch_labels))

        test_accuracy = calc_accuracy(test_prediction.eval(), test_labels)
        logger.info('Test accuracy: %.1f%%', test_accuracy)

def train_perceptron(
        train_dataset, train_labels, test_dataset, test_labels, reg_coeff
):
    """Basic MLP with one hidden layer and nonlinearity. Trying some regularization techniques."""
    batch_size = 128
    num_samples = train_dataset.shape[0]

    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(52)

        input_data = tf.placeholder(tf.float32, shape=(None, IMAGE_RES * IMAGE_RES))
        input_labels = tf.placeholder(tf.float32, shape=(None, NUM_CLASSES))
        dropout_keep_prob = tf.placeholder(tf.float32)

        w_1 = tf.Variable(tf.truncated_normal([IMAGE_RES * IMAGE_RES, 1024]))
        b_1 = tf.Variable(tf.zeros([1024]))
        layer_1 = tf.nn.relu(tf.matmul(input_data, w_1) + b_1)
        layer_1 = tf.nn.dropout(layer_1, dropout_keep_prob)

        w_2 = tf.Variable(tf.truncated_normal([1024, NUM_CLASSES]))
        b_2 = tf.Variable(tf.zeros(NUM_CLASSES))
        logits = tf.matmul(layer_1, w_2) + b_2

        unregularized_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=input_labels, logits=logits),
        )

        l2_loss = reg_coeff * (l2(w_1) + l2(w_2))

        loss = unregularized_loss + l2_loss

        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
        prediction_op = tf.nn.softmax(logits)

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        logger.info('Initialized')
        for step in range(5001):
            offset = (step * batch_size) % (num_samples - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]

            feed_dict = {input_data: batch_data, input_labels: batch_labels, dropout_keep_prob: 0.5}
            _, l, predictions = sess.run([optimizer, loss, prediction_op], feed_dict=feed_dict)

            if step % 500 == 0:
                logger.info('Minibatch loss at step %d is %f', step, l)
                logger.info('Minibatch accuracy: %.1f%%', calc_accuracy(predictions, batch_labels))

                def evaluate(data, labels):
                    """Calculate accuracy for dataset."""
                    predictions = sess.run(
                        prediction_op, feed_dict={input_data: data, dropout_keep_prob: 1.0}
                    )
                    return calc_accuracy(predictions, labels)

                train_acc = evaluate(train_dataset, train_labels)
                test_acc = evaluate(test_dataset, test_labels)
                logger.info('Train accuracy: %.1f%%', train_acc)
                logger.info('Test accuracy: %.1f%%', test_acc)

def train_better(
        train_dataset, train_labels, test_dataset, test_labels, reg_coeff=0.000005
):
    """
    Trying more regularization stuff and a different optimizer.
    This one is able to achieve 90.8% on unsanitized testing dataset with just one hidden layer.

    """
    batch_size = 128
    num_samples = train_dataset.shape[0]

    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(52)

        input_data = tf.placeholder(tf.float32, shape=(None, IMAGE_RES * IMAGE_RES))
        input_labels = tf.placeholder(tf.float32, shape=(None, NUM_CLASSES))
        dropout_keep_prob = tf.placeholder(tf.float32)

        w_1 = tf.Variable(tf.truncated_normal([IMAGE_RES * IMAGE_RES, 1024]))
        b_1 = tf.Variable(tf.zeros([1024]))
        layer_1 = tf.nn.relu(tf.matmul(input_data, w_1) + b_1)
        layer_1 = tf.nn.dropout(layer_1, dropout_keep_prob)

        w_2 = tf.Variable(tf.truncated_normal([1024, NUM_CLASSES]))
        b_2 = tf.Variable(tf.zeros(NUM_CLASSES))
        logits = tf.matmul(layer_1, w_2) + b_2

        prediction_op = tf.nn.softmax(logits)

        unregularized_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=input_labels, logits=logits),
        )

        l2_loss = reg_coeff * (l2(w_1) + l2(w_2))
        loss = unregularized_loss + l2_loss

        optimizer = tf.train.AdamOptimizer(learning_rate=(1e-4), epsilon=1e-3).minimize(loss)

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        logger.info('Initialized')
        for step in range(100000000):
            offset = (step * batch_size) % (num_samples - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]

            feed_dict = {input_data: batch_data, input_labels: batch_labels, dropout_keep_prob: 0.5}
            sess.run(optimizer, feed_dict=feed_dict)

            if step % 1000 == 0:
                l, ureg_l, l2l, predictions = sess.run(
                    [loss, unregularized_loss, l2_loss, prediction_op],
                    feed_dict=feed_dict,
                )
                logger.info('Minibatch loss at step %d is %f %f %f', step, l, ureg_l, l2l)
                logger.info('Minibatch accuracy: %.1f%%', calc_accuracy(predictions, batch_labels))

                def evaluate(data, labels):
                    """Calculate accuracy for dataset."""
                    predictions, l = sess.run(
                        [prediction_op, loss],
                        feed_dict={input_data: data, input_labels: labels, dropout_keep_prob: 1}
                    )
                    return calc_accuracy(predictions, labels), l

                train_acc, train_loss = evaluate(train_dataset, train_labels)
                test_acc, test_loss = evaluate(test_dataset, test_labels)
                logger.info('Train accuracy: %.1f%% loss: %f', train_acc, train_loss)
                logger.info('Test accuracy: %.1f%% loss: %f', test_acc, test_loss)

def train_deeper(train_dataset, train_labels, test_dataset, test_labels):
    """
    Training perceptron with more hidden layers.
    This requires more regularization techniques, especially xavier weight initialization and
    batch normalization (as well as weight decay and dropout).

    """
    batch_size = 256
    num_samples = train_dataset.shape[0]

    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(52)

        x = tf.placeholder(tf.float32, shape=(None, IMAGE_RES * IMAGE_RES), name='x')
        y = tf.placeholder(tf.float32, shape=(None, NUM_CLASSES), name='y')
        keep_prob = tf.placeholder(tf.float32)
        is_training = tf.placeholder(tf.bool, name='is_training')

        fc1 = dense_batch_relu_dropout(x, 1024, is_training, keep_prob, 'fc1')
        fc2 = dense_batch_relu_dropout(fc1, 300, is_training, keep_prob, 'fc2')
        fc3 = dense_batch_relu_dropout(fc2, 50, is_training, keep_prob, 'fc3')
        logits = dense(fc3, NUM_CLASSES, 'logits')

        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1)), tf.float32),
            )

        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            train_step = tf.train.AdamOptimizer(learning_rate=1e-5, epsilon=1e-3).minimize(loss)

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        logger.info('Initialized')
        for step in range(100000000):
            offset = (step * batch_size) % (num_samples - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]

            feed = {x: batch_data, y: batch_labels, keep_prob: 0.5, is_training: True}
            sess.run(train_step, feed_dict=feed)

            if step % 1000 == 0:
                np.set_printoptions(precision=3)

                test_feed = {x: batch_data, y: batch_labels, keep_prob: 0.5, is_training: False}
                ac, l = sess.run([accuracy, loss], feed_dict=test_feed)
                logger.info('Minibatch loss at step %d is %f', step, l)
                logger.info('Minibatch accuracy: %.2f%%', ac * 100)

                sample_logits = sess.run(
                    logits, feed_dict={x: batch_data[:1], keep_prob: 1, is_training: False},
                )
                logger.info(
                    'Sample logits: %r mean: %f',
                    sample_logits, np.mean(np.absolute(sample_logits)),
                )

                train_ac, train_l = sess.run(
                    [accuracy, loss],
                    feed_dict={x: train_dataset, y: train_labels, keep_prob: 1, is_training: False},
                )
                test_ac, test_l = sess.run(
                    [accuracy, loss],
                    feed_dict={x: test_dataset, y: test_labels, keep_prob: 1, is_training: False},
                )
                logger.info('Train accuracy: %.2f%% loss: %f', train_ac * 100, train_l)
                logger.info('Test accuracy: %.2f%% loss: %f', test_ac * 100, test_l)

def main():
    """Script entry point."""
    init_logger(os.path.basename(__file__))
    tf.logging.set_verbosity(logging.WARN)

    args = parse_args()
    logger.info('Args: %r', args)

    pickle_filename = PICKLE_FILE_SANITIZED if args.sanitized else PICKLE_FILE
    datasets = load_datasets(pickle_filename)

    train_dataset, train_labels = get_flattened_dataset(datasets, 'train')
    test_dataset, test_labels = get_flattened_dataset(datasets, 'valid')
    valid_dataset, valid_labels = get_flattened_dataset(datasets, 'test')

    logger.info('%r %r', train_dataset.shape, train_labels.shape)
    logger.info('%r %r', test_dataset.shape, test_labels.shape)
    logger.info('%r %r', valid_dataset.shape, valid_labels.shape)

    train_deeper(train_dataset, train_labels, test_dataset, test_labels)

    return 0


if __name__ == '__main__':
    sys.exit(main())
