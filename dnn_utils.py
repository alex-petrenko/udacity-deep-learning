import numpy as np
import tensorflow as tf

from tensorflow.nn import l2_loss as l2


def calc_accuracy(predictions, labels):
    total_correct = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    accuracy = (total_correct * 100.0) / predictions.shape[0]
    return accuracy

def dense(x, size, scope):
    return tf.contrib.layers.fully_connected(x, size, activation_fn=None, scope=scope)

# def dense_batch_relu_dropout1(x, size, is_training, keep_prob, scope):
#     with tf.variable_scope(scope):
#         w = tf.Variable(tf.truncated_normal([784, size]))
#         b = tf.Variable(tf.zeros([size]))
#         fc = tf.nn.relu(tf.matmul(x, w) + b)
#         fc_norm = tf.contrib.layers.batch_norm(
#             fc, center=True, scale=True, is_training=is_training, scope='bn',
#         )
#         fc_relu = tf.nn.relu(fc, 'relu')
#         return tf.nn.dropout(fc_relu, keep_prob)

def dense_batch_relu_dropout(x, size, is_training, keep_prob, scope):
    with tf.variable_scope(scope):
        fc = dense(x, size, 'dense')
        fc_norm = tf.contrib.layers.batch_norm(
            fc, center=True, scale=True, is_training=is_training, scope='bn',
        )
        fc_relu = tf.nn.relu(fc_norm, 'relu')
        return tf.nn.dropout(fc_relu, keep_prob)