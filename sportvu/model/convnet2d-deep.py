from __future__ import division
import tensorflow as tf
import numpy as np
from utils import *

# play with were batch norm is (2 conv layers before batch norm - no first layer bn)
# play with kernel size of conv layers (3x3 kernel instead of 5x5)
# player with size of conv dims(larger) and dims of fc layers(smaller)
# play with linearities
# play with optimization functions

# build lstm model

class ConvNet2dDeep:
    """

    """

    def __init__(self, config):
        self.config = config
        self.d1 = config['d1']
        self.d2 = config['d2']
        self.conv_blocks = config['conv_blocks']
        self.fc_layers = config['fc_layers']
        self.value_keep_prob = config['keep_prob']
        self.activation = config['activation']
        if 'bn' in config:## later version
            self.bn = config['bn']
            if self.bn:
                self.training = tf.placeholder(tf.bool)
                self.bn_inds = config['bn_inds']
        else:
            self.bn = False
        if 'pool' in config:## later version
            self.pool = config['pool']
        else:
            self.pool = False
    # convnet

    def build(self):
        # placeholders
        x = tf.placeholder(tf.float32, [None, self.d1, self.d2, 3])
        keep_prob = tf.placeholder(tf.float32)

        # init weights/bias
        # conv
        W_conv = []
        b_conv = []
        for block_ind in xrange(len(self.conv_blocks)):
            for layer_ind in xrange(len(self.conv_blocks[block_ind])):
                W_conv.append(weight_variable(self.conv_blocks[block_ind][layer_ind]))
                b_conv.append(bias_variable(self.conv_blocks[block_ind][layer_ind][-1]))
        if self.pool:
            SHAPE_convlast = int(np.ceil(self.d1 / (2**len(self.conv_blocks))) *
                                 np.ceil(self.d2 / (2**len(self.conv_blocks))) *
                                 self.conv_blocks[-1][-1][-1])
        else:
            SHAPE_convlast = int(np.ceil(self.d1 ) *
                                 np.ceil(self.d2 ) *
                                 self.conv_blocks[-1][-1][-1])
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
        for block_ind in xrange(len(self.conv_blocks)):
            for layer_ind in xrange(len(self.conv_blocks[block_ind])):
                if self.activation == 'ReLU':
                    h_conv = tf.nn.relu(
                        conv2d(h_pool_drop, W_conv[block_ind * self.block_size  + layer_ind]) + b_conv[block_ind * self.block_size  + layer_ind]
                    )
                else:
                    h_conv = leaky_ReLU(
                        conv2d(h_pool_drop, W_conv[block_ind * self.block_size  + layer_ind]) + b_conv[block_ind * self.block_size  + layer_ind]
                    )
            if self.pool:
                h_pool = max_pool_2x2(h_conv)
            else:
                h_pool = h_conv
            if self.bn and layer_ind in self.bn_inds:
                h_pool = bn(h_pool, self.training)
            h_pool_drop = tf.nn.dropout(h_pool, keep_prob)
        # fc
        h_pool_flat = tf.reshape(h_pool_drop, [-1, SHAPE_convlast])
        h_fc_drop = h_pool_flat
        for layer_ind in xrange(len(self.fc_layers) - 2):
            if self.activation == 'ReLU':
                h_fc = tf.nn.relu(
                    tf.matmul(h_fc_drop, W_fc[layer_ind]) + b_fc[layer_ind]
                )
            else:
                h_fc = leaky_ReLU(
                    tf.matmul(h_fc_drop, W_fc[layer_ind]) + b_fc[layer_ind]
                )
            if self.bn:
                h_fc = bn(h_fc, self.training)
            h_fc_drop = tf.nn.dropout(h_fc, keep_prob)

        y = tf.matmul(h_fc_drop, W_fc[-1]) + b_fc[-1]

        self.x = x
        self.keep_prob = keep_prob
        self.logits = y

    def input(self, x, keep_prob=None, training=False):
        if keep_prob == None: #default, 'training'
            keep_prob = self.value_keep_prob
        ret_dict = {}
        ret_dict[self.x] = x
        ret_dict[self.keep_prob] = keep_prob
        if self.bn:
            ret_dict[self.training] = training
        return ret_dict

    def output(self):
        return self.logits


if __name__ == '__main__':
    # import numpy as np

    net_config = {
        'd1': 100,
        'd2': 50,
        'conv_blocks': [[[5, 5, 3, 32], [5, 5, 32, 32]], [[5, 5, 32, 64], [5, 5, 64, 64]]],
        'fc_layers': [1024, 2],
    }
    net = ConvNet2dDeep(net_config)

    batch_xs = np.random.rand(32, 100, 50, 3)
    batch_ys = np.array([[0, 1]]).repeat(32, axis=0)

    net.build()

    # build loss
    y_ = tf.placeholder(tf.float32, [None, 2])
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=net.output()))
    train_step = tf.train.AdagradOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(net.output(), 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    feed_dict = net.input(batch_xs, .5)
    feed_dict[y_] = batch_ys
    train_accuracy = accuracy.eval(feed_dict=feed_dict)