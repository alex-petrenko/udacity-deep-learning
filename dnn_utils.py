"""
Various helpers for NN training.

"""


import numpy as np
import tensorflow as tf


def l2(t):
    """l2 norm alias."""
    return tf.nn.l2_loss(t)

def calc_accuracy(predictions, labels):
    total_correct = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    accuracy = (total_correct * 100.0) / predictions.shape[0]
    return accuracy

def dense(x, size, regularizer, scope):
    return tf.contrib.layers.fully_connected(
        x,
        size,
        activation_fn=None,
        scope=scope,
        weights_regularizer=regularizer,
        biases_regularizer=regularizer,
    )

def dense_regularized(
        x, size, is_training, keep_prob, regularizer, scope, activation=tf.nn.relu,
):
    with tf.variable_scope(scope):
        fc = dense(x, size, regularizer, 'dense')
        fc_norm = tf.contrib.layers.batch_norm(
            fc,
            center=True,
            scale=True,
            is_training=is_training,
            fused=True,
            zero_debias_moving_mean=True,
            scope='bn',
        )
        fc_act = activation(fc_norm, 'activation')
        return tf.nn.dropout(fc_act, keep_prob)

def conv_layer(x, num_outputs, kernel_size, stride, regularizer, scope, padding='SAME'):
    return tf.contrib.layers.conv2d(
        x,
        num_outputs,
        kernel_size,
        stride=stride,
        padding=padding,
        activation_fn=None,
        weights_regularizer=regularizer,
        biases_regularizer=regularizer,
        scope=scope,
    )

def conv_layer_regularized(
    x,
    num_outputs,
    kernel_size,
    stride,
    regularizer,
    is_training,
    keep_prob,
    scope,
    padding='SAME',
    activation=tf.nn.relu):
    """Conv2d layer with batch norm, activation and dropout."""
    with tf.variable_scope(scope):
        conv = conv_layer(x, num_outputs, kernel_size, stride, regularizer, 'conv', padding)
        conv_norm = tf.contrib.layers.batch_norm(
            conv,
            center=True,
            scale=True,
            is_training=is_training,
            fused=True,
            zero_debias_moving_mean=True,
            scope='bn',
        )
        conv_act = activation(conv_norm, 'activation')
        return tf.nn.dropout(conv_act, keep_prob)

def layer_summaries(tensor, scope):
    """Add basic summaries for 1-dimensional tensors."""
    with tf.name_scope(scope):
        mean = tf.reduce_mean(tensor)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(tensor - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(tensor))
        tf.summary.scalar('min', tf.reduce_min(tensor))
        tf.summary.histogram('histogram', tensor)

def count_total_parameters():
    """
    Returns total number of trainable parameters in the current tf graph.
    https://stackoverflow.com/a/38161314/1645784

    """
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters

def visualize_filters(layer_name):
    """Add filters to Tensorboard."""
    with tf.variable_scope(layer_name, reuse=True):
        kernel = tf.get_variable('weights')
        with tf.variable_scope('visualization'):
            x_min = tf.reduce_min(kernel)
            x_max = tf.reduce_max(kernel)
            kernel_0_to_1 = (kernel - x_min) / (x_max - x_min)
            # to tf.image_summary format [batch_size, height, width, channels]
            kernel_transposed = tf.transpose(kernel_0_to_1, [3, 0, 1, 2])
            # this will display random 3 filters from the 64 in conv1
            tf.summary.image(layer_name + '/filters', kernel_transposed, max_outputs=8)
