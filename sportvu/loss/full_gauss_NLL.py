from __future__ import division
import tensorflow as tf
from tensorflow.python.ops.distributions.util import fill_lower_triangular


class FullGaussNLL(object):
    def __init__(self):
        super(FullGaussNLL, self).__init__()

    def build_tf_loss(self, pred, y_, eps=1e-5):
        """
        only works if #players=11
        k = 22
        y_ shape (N, T, k)
        pred shape (N, T, k+k*(k-1)/2)

        """
        pred_dim = y_.get_shape()[-1].value
        mean, R = tf.split(pred, [int(pred_dim), int(pred_dim * (pred_dim + 1) / 2)], axis=2)  # Sigma_inv = RR^T Cholasky decomp
        R = fill_lower_triangular(tf.nn.softplus(R) + eps)

        diff_shape = [s.value for s in y_.shape]
        diff_shape.append(1)
        diff = tf.reshape(mean - y_, diff_shape)
        diff_loss = tf.pow(tf.matmul(tf.transpose(R,[0,1,3,2], diff), 2))
        diff_loss = tf.reduce_mean(tf.reduce_sum(diff_loss,axis=2))
        normalizer_loss = tf.log(tf.matrix_diag_part(R))
        normalizer_loss = tf.reduce_mean(tf.reduce_sum(normalizer_loss, axis=2))
        return diff_loss - normalizer_loss  # +CONST
