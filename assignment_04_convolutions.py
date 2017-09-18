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
        'regularization_coeff': 0.00005,
        'keep_prob': 0.5,
        'batch_size': 128,
        'first_conv': 5,
        'fc_size': 1024,
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
    first_conv = int(get_param('first_conv'))
    fc_size = int(get_param('fc_size'))
    activation_func = activation_funcs[get_param('activation')]

    # training parameters
    save_restore = False
    time_limit_seconds = 3600 * 2

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
            return tf.contrib.layers.conv2d(
                x,
                size,
                kernel_size,
                stride=stride,
                padding='SAME',
                activation_fn=activation_func,
                weights_regularizer=regularizer,
                biases_regularizer=regularizer,
                scope=name,
            )
        def batch_norm(x, name):
            return tf.contrib.layers.batch_norm(
                x, center=True, scale=True, is_training=is_training, fused=True, scope=name,
            )

        conv1 = conv(x, 32, (first_conv, first_conv), 1, 'conv1')
        bn1 = batch_norm(conv1, 'bn1')
        conv2 = conv(bn1, 32, (first_conv, first_conv), 1, 'conv2')
        bn2 = batch_norm(conv2, 'bn2')
        pool1 = tf.contrib.layers.max_pool2d(bn2, kernel_size=2)  # 14x14
        drop1 = tf.nn.dropout(pool1, keep_prob)
        conv3 = conv(drop1, 64, (3, 3), 1, 'conv3')
        bn3 = batch_norm(conv3, 'bn3')
        conv4 = conv(bn3, 64, (3, 3), 1, 'conv4')
        bn4 = batch_norm(conv4, 'bn4')
        pool2 = tf.contrib.layers.max_pool2d(bn4, kernel_size=2)  # 7x7
        drop2 = tf.nn.dropout(pool2, keep_prob)
        flatten = tf.contrib.layers.flatten(drop2)
        fc1 = fully_connected(flatten, fc_size, 'fc1')
        fc2 = fully_connected(fc1, fc_size, 'fc2')
        logits = dense(fc2, NUM_CLASSES, regularizer, 'logits')

        logger.info('Total parameters in the model: %d', count_total_parameters())

        visualize_filters('conv1')
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
            optimizer = tf.train.AdamOptimizer(learning_rate=(1e-4), epsilon=1e-4)
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
                    sess.run(
                        train_op,
                        feed_dict={keep_prob: keep_prob_param, is_training: True},
                    )

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

    param_grid = {
        'regularization_coeff': (0.0005, 0.00001, 0.00005, 0.00001, 0.000005),
        'batch_size': (64, 128),
        'first_conv': (3, 5),
        'fc_size': (512, 1024, 2048),
    }

    initial_guess = {
        'regularization_coeff': 0.00005,
        'batch_size': 128,
        'first_conv': 5,
        'fc_size': 1024,
    }

    logger.info('Grid: %r', param_grid)
    def train_func(params):
        logger.info('Training %r', params)
        try:
            ac = train_convnet(train_dataset, train_labels, test_dataset, test_labels, params)
            return ac
        except KeyboardInterrupt:
            logger.info('Keyboard interrupt!')
            raise
        except Exception as exc:
            logger.info('Exception %r', exc)
            return -1000

    from hyperopt import EvolutionaryHyperopt
    hyperopt = EvolutionaryHyperopt()
    hyperopt.set_param_grid(param_grid)
    hyperopt.set_initial_guess(initial_guess)
    hyperopt.set_evaluation_func(train_func)
    hyperopt.set_checkpoint_dir('.hyperopt.checkpoints')
    hyperopt.try_initialize_from_checkpoint()

    try:
        best = hyperopt.optimize()
        logger.info(best)
    except KeyboardInterrupt:
        logger.info('Terminated!')
    finally:
        hyperopt.log_halloffame()

    return 0


if __name__ == '__main__':
    sys.exit(main())
