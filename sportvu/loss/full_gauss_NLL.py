import tensorflow as tf


class full_gauss_NLL(object):
    def __init__(self):
        super(full_gauss_NLL, self).__init__()

    def build_tf_loss(self, pred, y_, eps=1e-5):
        """
        only works if #players=11
        k = 22
        y_ shape (N, T, k)
        pred shape (N, T, k+k*(k-1)/2)

        """
        pred_dim = 22
        mean, R_trans = tf.split(pred, [pred_dim, pred_dim * (pred_dim + 1) / 2],
                                 axis=-1)  # Sigma_inv = R^TR cholasky upper decomp
        R_trans = fill_lower_triangular(tf.nn.softplus(R_trans) + eps)

        diff_shape = [s.value for s in gt.shape].append(1)
        diff = tf.reshape(mean - gt, diff_shape)
        diff_loss = tf.reduce_mean(tf.pow(tf.matmul(R_trans, diff), 2))
        normalizer_loss = tf.reduce_mean(tf.log(tf.matrix_diag_part(R_trans)))
        return diff_loss + normalizer_loss  # +CONST
