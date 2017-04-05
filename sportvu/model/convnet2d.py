from __future__ import division
import tensorflow as tf
import numpy as np


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


class ConvNet2d:
    """

    """

    def __init__(self, config):
        self.config = config
        self.d1 = config['d1']
        self.d2 = config['d2']
        self.conv_layers = config['conv_layers']
        self.fc_layers = config['fc_layers']
    # convnet

    def build(self):
        # placeholders
        x = tf.placeholder(tf.float32, [None, self.d1, self.d2, 3])
        keep_prob = tf.placeholder(tf.float32)

        # init weights/bias
        # conv
        W_conv = []
        b_conv = []
        for layer_ind in xrange(len(self.conv_layers)):
            W_conv.append(weight_variable(self.conv_layers[layer_ind]))
            b_conv.append(bias_variable([self.conv_layers[layer_ind][-1]]))

        SHAPE_convlast = int(np.ceil(self.d1 / (2**len(self.conv_layers))) *
                             np.ceil(self.d2 / (2**len(self.conv_layers))) *
                             self.conv_layers[-1][-1])
        # fc
        W_fc = []
        b_fc = []
        self.fc_layers.insert(0, SHAPE_convlast)
        for layer_ind in xrange(len(self.fc_layers) - 1):
            W_fc.append(weight_variable(
                [self.fc_layers[layer_ind], self.fc_layers[layer_ind + 1]]))
            b_fc.append(bias_variable([self.fc_layers[layer_ind + 1]]))

        # build model
        # conv
        h_pool_drop = x
        for layer_ind in xrange(len(self.conv_layers)):
            h_conv = tf.nn.relu(
                conv2d(h_pool_drop, W_conv[layer_ind]) + b_conv[layer_ind])
            h_pool = max_pool_2x2(h_conv)
            h_pool_drop = tf.nn.dropout(h_pool, keep_prob)
        # fc
        h_pool_flat = tf.reshape(h_pool_drop, [-1, SHAPE_convlast])
        h_fc_drop = h_pool_flat
        for layer_ind in xrange(len(self.fc_layers) - 2):
            h_fc = tf.nn.relu(tf.matmul(h_pool_flat, W_fc[
                              layer_ind]) + b_fc[layer_ind])
            h_fc_drop = tf.nn.dropout(h_fc, keep_prob)

        y = tf.matmul(h_fc_drop, W_fc[-1]) + b_fc[-1]

        self.x = x
        self.keep_prob = keep_prob
        self.logits = y

    def input(self, x, keep_prob):
        return {self.x: x,
                self.keep_prob: keep_prob}

    def output(self):
        return self.logits


if __name__ == '__main__':
    # import numpy as np

    net_config = {
        'd1': 100,
        'd2': 50,
        'conv_layers': [[5, 5, 3, 32], [5, 5, 32, 64]],
        'fc_layers': [1024, 2],
    }
    net = ConvNet2d(net_config)

    batch_xs = np.random.rand(32, 100, 50, 3)
    batch_ys = np.array([[0, 1]]).repeat(32, axis=0)

    net.build()

    # build loss
    y_ = tf.placeholder(tf.float32, [None, 2])
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=net.output()))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(net.output(), 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    feed_dict = net.input(batch_xs, .5)
    feed_dict[y_] = batch_ys
    train_accuracy = accuracy.eval(feed_dict=feed_dict)
