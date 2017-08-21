import tensorflow as tf


class diag_gauss_NLL(object):
    def __init__(self):
        super(diag_gauss_NLL, self).__init__()

    def build_tf_loss(self, pred, y_,eps=1e-5):
        """
        y_ shape (N, T, 2*#players)
        pred shape (N, T, 4*#players)
        """
        mean, pre_var = tf.split(pred, 2, axis=-1)
        var = tf.nn.softplus(pre_std) + eps
        return tf.reduce_mean(0.5 * tf.div(tf.pow(mean - gt, 2), var) - 0.5 * tf.log(var))  # +const