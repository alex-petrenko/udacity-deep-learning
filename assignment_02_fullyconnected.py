import os
import sys
import pickle
import logging
import argparse
import numpy as np
import tensorflow as tf

from utils import *
from dataset_utils import *


logger = logging.getLogger(os.path.basename(__file__))


def parse_args():
    """Parse command line args using argparse."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sanitized',
        action='store_true',
        help='Use sanitized version of the test and validation datasets',
    )
    return parser.parse_args()

def flatten_dataset(data, labels):
    assert data.ndim == 3
    assert data.shape[1] == data.shape[2]
    data = data.reshape((-1, data.shape[1] * data.shape[2])).astype(np.float32)
    labels = (np.arange(NUM_CLASSES) == labels[:, None]).astype(np.float32)
    return data, labels

def calc_accuracy(predictions, labels):
    total_correct = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    accuracy = (total_correct * 100.0) / predictions.shape[0]
    return accuracy

def train_gradient_descent(train_dataset, train_labels, test_dataset, test_labels):
    train_subset = 10000

    graph = tf.Graph()
    with graph.as_default():
        tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
        tf_train_labels = tf.constant(train_labels[:train_subset])
        tf_test_dataset = tf.constant(test_dataset)

        weights = tf.Variable(tf.truncated_normal([IMAGE_RES * IMAGE_RES, NUM_CLASSES]))
        biases = tf.Variable(tf.zeros([NUM_CLASSES]))
        logits = tf.matmul(tf_train_dataset, weights) + biases
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits),
        )
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        train_prediction = tf.nn.softmax(logits)
        test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        logger.info('Initialized')
        for step in range(801):
            _, total_loss, predictions = sess.run([optimizer, loss, train_prediction])
            if step % 100 == 0:
                logger.info('Loss at step %d is %f', step, total_loss)
                accuracy = calc_accuracy(predictions, train_labels[:train_subset, :])
                logger.info('Training accuracy: %f', accuracy)

        test_accuracy = calc_accuracy(test_prediction.eval(), test_labels)
        logger.info('Test accuracy: %f', test_accuracy)

def train_stochastic_gradient_descent(train_dataset, train_labels, test_dataset, test_labels):
    batch_size = 128

    graph = tf.Graph()
    with graph.as_default():
        tf_train_dataset = tf.placeholder(tf.float32, shape=(None, IMAGE_RES * IMAGE_RES))
        tf_train_labels = tf.placeholder(tf.float32, shape=(None, NUM_CLASSES))
        tf_test_dataset = tf.constant(test_dataset)

        weights = tf.Variable(tf.truncated_normal([IMAGE_RES * IMAGE_RES, NUM_CLASSES]))
        biases = tf.Variable(tf.zeros([NUM_CLASSES]))
        logits = tf.matmul(tf_train_dataset, weights) + biases
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits),
        )

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

            if step % 500 == 0:
                logger.info('Minibatch loss at step %d is %f', step, l)
                logger.info('Minibatch accuracy: %.1f%%', calc_accuracy(predictions, batch_labels))

        test_accuracy = calc_accuracy(test_prediction.eval(), test_labels)
        logger.info('Test accuracy: %.1f%%', test_accuracy)

def train_perceptron(train_dataset, train_labels, test_dataset, test_labels):
    batch_size = 128

    graph = tf.Graph()
    with graph.as_default():
        input_data = tf.placeholder(tf.float32, shape=(None, IMAGE_RES * IMAGE_RES))
        input_labels = tf.placeholder(tf.float32, shape=(None, NUM_CLASSES))

        w_1 = tf.Variable(tf.truncated_normal([IMAGE_RES * IMAGE_RES, 1024]))
        b_1 = tf.Variable(tf.zeros([1024]))
        layer_1 = tf.nn.relu(tf.matmul(input_data, w_1) + b_1)

        w_2 = tf.Variable(tf.truncated_normal([1024, NUM_CLASSES]))
        b_2 = tf.Variable(tf.zeros(NUM_CLASSES))
        logits = tf.matmul(layer_1, w_2) + b_2

        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=input_labels, logits=logits),
        )

        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
        prediction_op = tf.nn.softmax(logits)

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        logger.info('Initialized')
        for step in range(30001):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]

            feed_dict = {input_data: batch_data, input_labels: batch_labels}
            _, l, predictions = sess.run([optimizer, loss, prediction_op], feed_dict=feed_dict)

            if step % 500 == 0:
                logger.info('Minibatch loss at step %d is %f', step, l)
                logger.info('Minibatch accuracy: %.1f%%', calc_accuracy(predictions, batch_labels))

                def evaluate(data, labels):
                    predictions = sess.run(prediction_op, feed_dict={input_data: data})
                    return calc_accuracy(predictions, labels)

                train_acc = evaluate(train_dataset, train_labels)
                test_acc = evaluate(test_dataset, test_labels)
                logger.info('Train accuracy: %.1f%%', train_acc)
                logger.info('Test accuracy: %.1f%%', test_acc)

def main():
    """Script entry point."""
    init_logger(os.path.basename(__file__))
    args = parse_args()
    logger.info('Args: %r', args)

    pickle_filename = PICKLE_FILE_SANITIZED if args.sanitized else PICKLE_FILE
    datasets = load_datasets(pickle_filename)

    train_dataset, train_labels = extract_dataset(datasets, 'train')
    test_dataset, test_labels = extract_dataset(datasets, 'valid')
    valid_dataset, valid_labels = extract_dataset(datasets, 'test')

    del datasets

    train_dataset, train_labels = flatten_dataset(train_dataset, train_labels)
    test_dataset, test_labels = flatten_dataset(test_dataset, test_labels)
    valid_dataset, valid_labels = flatten_dataset(valid_dataset, valid_labels)

    logger.info('%r %r', train_dataset.shape, train_labels.shape)
    logger.info('%r %r', test_dataset.shape, test_labels.shape)
    logger.info('%r %r', valid_dataset.shape, valid_labels.shape)

    train_perceptron(train_dataset, train_labels, test_dataset, test_labels)

    return 0


if __name__ == '__main__':
    sys.exit(main())