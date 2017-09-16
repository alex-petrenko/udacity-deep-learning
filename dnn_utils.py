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

def dense_batch_relu_dropout(
        x, size, is_training, keep_prob, regularizer, scope, activation=tf.nn.relu,
):
    with tf.variable_scope(scope):
        fc = dense(x, size, regularizer, 'dense')
        fc_norm = tf.contrib.layers.batch_norm(
            fc, center=True, scale=True, is_training=is_training, fused=True, scope='bn',
        )
        fc_act = activation(fc_norm, 'activation')
        return tf.nn.dropout(fc_act, keep_prob)
