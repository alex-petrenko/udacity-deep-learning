"""
Udacity Deep Learning course by Google.
Assignment #03: various regularization techniques.

"""

import os
import sys
import time
import logging
import argparse

from os.path import join

import numpy as np
import tensorflow as tf

from tensorflow.contrib.data import Dataset

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

def train_logistic_classifier(train_dataset, train_labels, test_dataset, test_labels, params):
    """Standard logistic classifier with no nonlinearity."""
    batch_size = 128
    learning_rate = params.get('learning_rate', 0.5)
    reg_coeff = params.get('weight_decay', 0.00001)

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

        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        train_prediction = tf.nn.softmax(logits)
        test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        for step in range(10001):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]

            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, predictions = sess.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

            verbose = True
            if step % 500 == 0:
                if verbose:
                    logger.info('Batch loss at step %d is %f', step, l)
                    logger.info('Batch accuracy: %.1f%%', calc_accuracy(predictions, batch_labels))

        test_accuracy = calc_accuracy(test_prediction.eval(), test_labels)
    return test_accuracy

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

        fc1 = dense_batch_relu_dropout(x, 1024, is_training, keep_prob, None, 'fc1')
        fc2 = dense_batch_relu_dropout(fc1, 300, is_training, keep_prob, None, 'fc2')
        fc3 = dense_batch_relu_dropout(fc2, 50, is_training, keep_prob, None, 'fc3')
        logits = dense(fc3, NUM_CLASSES, None, 'logits')

        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1)), tf.float32),
            )

        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            train_step = tf.train.AdamOptimizer(learning_rate=1e-3, epsilon=1e-3).minimize(loss)

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

def layer_summaries(tensor, scope):
    """Add basic summaries for 1-dimensional tensors."""
    logger.info('Adding summaries for tensor %r', tensor)
    with tf.name_scope(scope):
        mean = tf.reduce_mean(tensor)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(tensor - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(tensor))
        tf.summary.scalar('min', tf.reduce_min(tensor))
        tf.summary.histogram('histogram', tensor)

def train_deeper_better(train_data, train_labels, test_data, test_labels, params):
    """Same as 'train_deeper', but now with tf.contrib.data.Dataset input pipeline."""
    default_params = {
        'regularization_coeff': 0.00001,
        'keep_prob': 0.5,
        'batch_size': 128,
        'fc1_size': 2048,
        'fc2_size': 1024,
        'fc3_size': 1024,
        'fc4_size': 1024,
        'fc5_size': 512,
        'activation': 'tanh',
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

    regularization_coeff = get_param('regularization_coeff')
    keep_prob_param = get_param('keep_prob')
    batch_size = int(get_param('batch_size'))
    fc1_size = int(get_param('fc1_size'))
    fc2_size = int(get_param('fc2_size'))
    fc3_size = int(get_param('fc3_size'))
    fc4_size = int(get_param('fc4_size'))
    fc5_size = int(get_param('fc5_size'))
    activation_func = activation_funcs[get_param('activation')]

    save_restore = False
    time_limit_seconds = 3600

    saver_path = join(SAVER_FOLDER, train_deeper_better.__name__)

    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(52)

        global_step_tensor = tf.contrib.framework.get_or_create_global_step()
        epoch_tensor = tf.Variable(0, trainable=False, name='epoch')
        next_epoch = tf.assign_add(epoch_tensor, 1)

        # dataset definition
        dataset = Dataset.from_tensor_slices({'x': train_data, 'y': train_labels})
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_initializable_iterator()
        sample = iterator.get_next()
        x = sample['x']
        y = sample['y']

        # actual computation graph
        keep_prob = tf.placeholder(tf.float32)
        is_training = tf.placeholder(tf.bool, name='is_training')

        regularizer = tf.contrib.layers.l2_regularizer(scale=regularization_coeff)

        def fully_connected(x, size, name):
            return dense_batch_relu_dropout(
                x, size, is_training, keep_prob, regularizer, name, activation_func,
            )

        fc1 = fully_connected(x, fc1_size, 'fc1')
        fc2 = fully_connected(fc1, fc2_size, 'fc2')
        fc3 = fully_connected(fc2, fc3_size, 'fc3')
        fc4 = fully_connected(fc3, fc4_size, 'fc4')
        fc5 = fully_connected(fc4, fc5_size, 'fc5')
        logits = dense(fc5, NUM_CLASSES, regularizer, 'logits')

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

    test_accuracy = 0
    best_accuracy = 0
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

            unacceptable = [50, 60, 70, 80, 81, 82, 83, 84]
            for i, value in enumerate(unacceptable):
                if epoch > i and test_accuracy < value:
                    logger.info('Terminate early!')
                    return test_accuracy

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

    train_dataset, train_labels = get_flattened_dataset(datasets, 'train')
    test_dataset, test_labels = get_flattened_dataset(datasets, 'valid')
    valid_dataset, valid_labels = get_flattened_dataset(datasets, 'test')

    del datasets

    logger.info('%r %r', train_dataset.shape, train_labels.shape)
    logger.info('%r %r', test_dataset.shape, test_labels.shape)
    logger.info('%r %r', valid_dataset.shape, valid_labels.shape)

    # train_deeper_better(train_dataset, train_labels, test_dataset, test_labels, {})
    # return 0

    param_grid = {
        'regularization_coeff': np.logspace(-7, -2, num=20),
        'batch_size': (32, 64, 128, 256),
        'fc1_size': (128, 256, 512, 1024, 2048, 3000, 4000),
        'fc2_size': (128, 256, 512, 1024, 2048, 3000),
        'fc3_size': (128, 256, 512, 1024, 2048, 3000),
        'fc4_size': (128, 256, 512, 1024, 2048, 3000),
        'fc5_size': (128, 256, 512, 1024, 2048),
        'activation': ('relu', 'tanh'),
    }
    initial_guess = {
        'regularization_coeff': 0.00001,
        'batch_size': 128,
        'fc1_size': 2048,
        'fc2_size': 1024,
        'fc3_size': 1024,
        'fc4_size': 1024,
        'fc5_size': 512,
        'activation': 'tanh',
    }

    logger.info('Grid: %r', param_grid)
    def train_func(params):
        logger.info('Training %r', params)
        try:
            ac = train_deeper_better(train_dataset, train_labels, test_dataset, test_labels, params)
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
