"""
Udacity Deep Learning course by Google.
Assignment #04: convnets.

"""

import os
import sys
import time
import logging
import argparse

from os.path import join

import numpy as np
import tensorflow as tf

from utils import *
from dnn_utils import *
from dataset_utils import *


logger = logging.getLogger(os.path.basename(__file__))  # pylint: disable=invalid-name

SUMMARY_FOLDER = './.summary'
SAVER_FOLDER = './.sessions'


def parse_args():
    """Parse command line args using argparse."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sanitized',
        action='store_true',
        help='Use sanitized version of the test and validation datasets',
    )
    return parser.parse_args()

def train_convnet(train_data, train_labels, test_data, test_labels, params):
    """Convolutional neural net for notMNIST dataset."""
    default_params = {
        'regularization_coeff': 0.00002,
        'keep_prob': 0.5,
        'batch_size': 64,
        'activation': 'relu',
    }
    activation_funcs = {
        'relu': tf.nn.relu,
        'tanh': tf.nn.tanh,
    }
    def get_param(name):
        if name in params:
            return params[name]
        logger.warning('%s not found in param, use default value %r', name, default_params[name])
        return default_params[name]

    # model hyperparameters
    regularization_coeff = get_param('regularization_coeff')
    keep_prob_param = get_param('keep_prob')
    batch_size = int(get_param('batch_size'))
    activation_func = activation_funcs[get_param('activation')]

    # training parameters
    save_restore = True
    time_limit_seconds = None

    saver_path = join(SAVER_FOLDER, train_convnet.__name__)

    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(52)

        global_step_tensor = tf.contrib.framework.get_or_create_global_step()
        epoch_tensor = tf.Variable(0, trainable=False, name='epoch')
        next_epoch = tf.assign_add(epoch_tensor, 1)

        # dataset definition
        x, y, iterator = dataset_to_inputs(train_data, train_labels, batch_size)

        # actual computation graph
        keep_prob = tf.placeholder(tf.float32)
        is_training = tf.placeholder(tf.bool, name='is_training')
        regularizer = tf.contrib.layers.l2_regularizer(scale=regularization_coeff)

        def fully_connected(x, size, name):
            return dense_regularized(
                x, size, is_training, keep_prob, regularizer, name, activation_func,
            )
        def conv(x, size, kernel_size, stride, name):
            return conv_layer_regularized(
                x,
                size,
                kernel_size=kernel_size,
                stride=stride,
                regularizer=regularizer,
                is_training=is_training,
                keep_prob=keep_prob,
                scope=name,
                padding='SAME',
                activation=activation_func,
            )

        conv1 = conv(x, 16, (5, 5), 1, 'conv1')
        pool1 = tf.contrib.layers.max_pool2d(conv1, kernel_size=2)  # 14x14
        conv2 = conv(pool1, 32, (3, 3), 1, 'conv2')
        pool2 = tf.contrib.layers.max_pool2d(conv2, kernel_size=2)  # 7x7
        conv3 = conv(pool2, 64, (3, 3), 1, 'conv3')
        logger.info('Last conv layer: %r', conv3)
        fc1 = fully_connected(tf.contrib.layers.flatten(conv3), 1024, 'fc1')
        logger.info('Fully connected: %r', fc1)
        fc2 = fully_connected(fc1, 1024, 'fc2')
        logits = dense(fc2, NUM_CLASSES, regularizer, 'logits')

        layer_summaries(logits, 'logits_summaries')

        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1)), tf.float32),
            )
            accuracy_percent = 100 * accuracy
            tf.summary.scalar('accuracy_percent', accuracy_percent)

        with tf.name_scope('loss'):
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            regularization_loss = tf.reduce_sum(regularization_losses)
            cross_entropy_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y),
            )
            loss = cross_entropy_loss + regularization_loss
            tf.summary.scalar('regularization_loss', regularization_loss)
            tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)
            tf.summary.scalar('loss', loss)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # ensures that we execute the update_ops before performing the train_op
            # needed for batch normalization (apparently)
            optimizer = tf.train.AdamOptimizer(learning_rate=(1e-4), epsilon=1e-3)
            train_op = optimizer.minimize(loss, global_step=global_step_tensor)

        all_summaries = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(join(SUMMARY_FOLDER, 'train'))
        batch_writer = tf.summary.FileWriter(join(SUMMARY_FOLDER, 'batch'))
        test_writer = tf.summary.FileWriter(join(SUMMARY_FOLDER, 'test'))

        saver = tf.train.Saver(max_to_keep=3)

    test_accuracy, best_accuracy = 0, 0
    with tf.Session(graph=graph) as sess:
        restored = False
        if save_restore:
            try:
                saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir=SAVER_FOLDER))
                restored = True
            except ValueError as exc:
                logger.info('Could not restore previous session! %r', exc)
                logger.info('Starting from scratch!')

        if not restored:
            tf.global_variables_initializer().run()

        logger.info('Starting training...')
        start_time = time.time()
        def enough():
            if time_limit_seconds is None:
                return False
            elapsed = time.time() - start_time
            return elapsed > time_limit_seconds

        epoch = epoch_tensor.eval()
        new_epoch = True
        while not enough():
            logger.info('Starting new epoch #%d!', epoch)
            sess.run(iterator.initializer, feed_dict={})
            while not enough():
                step = tf.train.global_step(sess, tf.train.get_global_step())
                try:
                    sess.run(train_op, feed_dict={keep_prob: keep_prob_param, is_training: True})
                    if new_epoch:
                        new_epoch = False
                        l, reg_l, ac, summaries = sess.run(
                            [loss, regularization_loss, accuracy_percent, all_summaries],
                            feed_dict={keep_prob: keep_prob_param, is_training: False},
                        )
                        batch_writer.add_summary(summaries, global_step=step)
                        logger.info(
                            'Minibatch loss: %f, reg loss: %f, accuracy: %.2f%%',
                            l, reg_l, ac,
                        )
                except tf.errors.OutOfRangeError:
                    logger.info('End of epoch #%d', epoch)
                    break

            # end of epoch
            previous_epoch = epoch
            epoch = next_epoch.eval()
            new_epoch = True

            if previous_epoch % 5 == 0 and save_restore:
                saver.save(sess, saver_path, global_step=previous_epoch)

            def get_eval_dict(data, labels):
                """Data for evaluation."""
                return {x: data, y: labels, keep_prob: 1, is_training: False}

            train_l, train_ac, summaries = sess.run(
                [loss, accuracy_percent, all_summaries],
                feed_dict=get_eval_dict(train_data[:10000], train_labels[:10000]),
            )
            train_writer.add_summary(summaries, global_step=step)

            test_l, test_accuracy, summaries = sess.run(
                [loss, accuracy_percent, all_summaries],
                feed_dict=get_eval_dict(test_data, test_labels),
            )
            test_writer.add_summary(summaries, global_step=step)

            best_accuracy = max(best_accuracy, test_accuracy)

            logger.info('Train loss: %f, train accuracy: %.2f%%', train_l, train_ac)
            logger.info(
                'Test loss: %f, TEST ACCURACY: %.2f%%  BEST ACCURACY %.2f%%    <<<<<<<',
                test_l, test_accuracy, best_accuracy,
            )

    return best_accuracy


def main():
    """Script entry point."""
    init_logger(os.path.basename(__file__))
    np.set_printoptions(precision=3)
    tf.logging.set_verbosity(tf.logging.FATAL)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    args = parse_args()
    logger.info('Args: %r', args)

    if tf.gfile.Exists(SUMMARY_FOLDER):
        tf.gfile.DeleteRecursively(SUMMARY_FOLDER)
    tf.gfile.MakeDirs(SUMMARY_FOLDER)

    tf.gfile.MakeDirs(SAVER_FOLDER)

    pickle_filename = PICKLE_FILE_SANITIZED if args.sanitized else PICKLE_FILE
    datasets = load_datasets(pickle_filename)

    train_dataset, train_labels = get_image_dataset(datasets, 'train')
    test_dataset, test_labels = get_image_dataset(datasets, 'valid')
    valid_dataset, valid_labels = get_image_dataset(datasets, 'test')

    del datasets

    logger.info('%r %r', train_dataset.shape, train_labels.shape)
    logger.info('%r %r', test_dataset.shape, test_labels.shape)
    logger.info('%r %r', valid_dataset.shape, valid_labels.shape)

    train_convnet(train_dataset, train_labels, test_dataset, test_labels, {})

    return 0


if __name__ == '__main__':
    sys.exit(main())
