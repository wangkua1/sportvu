from __future__ import division
import tensorflow as tf
import numpy as np
from utils import *
from lstm_utils import *

class LSTM:
    """

    """

    def __init__(self, config):
        self.config = config
        self.d1 = config['d1']
        self.d2 = config['d2']
        self.conv_layers = config['conv_layers']
        self.fc_layers = config['fc_layers']
        self.value_keep_prob = config['keep_prob']
        self.batch_size = config['batch_size']
        self.channel_size = 3
        self.encoder_time_size = config['encoder_time_size']
        self.rnn_hid_dim = config['rnn_hid_dim']
        self.batch_size = config['batch_size']
        self.cell_type = config['cell_type']
        self.num_of_cells = config['num_of_cells']
        self.frame_rate = config['frame_rate']
        self.bn = config['bn']
        if self.bn:
            self.training = tf.placeholder(tf.bool)

    def build(self):
        # placeholders
        x = tf.placeholder(tf.float32, [self.batch_size, self.encoder_time_size * 2, self.d1, self.d2, self.channel_size])
        x = x[:, ::self.frame_rate]
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
        ## pre-rnn
        self.W_pre_rnn = weight_variable([SHAPE_convlast, self.rnn_hid_dim])
        self.b_pre_rnn = bias_variable([self.rnn_hid_dim])
        ## rnn
        # Enc
        if self.cell_type != 'BNLSTMCell':
            cell = eval('tf.contrib.rnn.%s' % (self.cell_type))(self.rnn_hid_dim, dropout_keep_prob=keep_prob) #, state_is_tuple=True)
        else:
            cell = BNLSTMCell(self.rnn_hid_dim, training=self.training)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
        enc_cell = tf.contrib.rnn.MultiRNNCell([cell] * self.num_of_cells)
        # fc (post-rnn)
        W_fc = []
        b_fc = []
        self.fc_layers.insert(0, self.rnn_hid_dim)
        for layer_ind in xrange(len(self.fc_layers) - 1):
            W_fc.append(weight_variable(
                [self.fc_layers[layer_ind], self.fc_layers[layer_ind + 1]]))
            b_fc.append(bias_variable([self.fc_layers[layer_ind + 1]]))

        ## Build Graph
        # build encoder
        # x (B,T,Y,X,C)
        tf_r_enc_input = x
        tf_r_enc_input = tf.reshape(tf_r_enc_input, # (B*T, Y,X,C)
                            (int(self.batch_size*self.encoder_time_size*2/self.frame_rate), self.d1,self.d2,self.channel_size))
        tf_r_enc_input = tf_r_enc_input
        # conv
        h_pool_drop = tf_r_enc_input
        for layer_ind in xrange(len(self.conv_layers)):
            if self.bn:
                h_conv = tf.nn.relu(bn(conv2d(h_pool_drop, W_conv[layer_ind]), self.training) + b_conv[layer_ind])
            else:
                h_conv = tf.nn.relu(conv2d(h_pool_drop, W_conv[layer_ind]) + b_conv[layer_ind])

            h_pool = max_pool_2x2(h_conv)
            h_pool_drop = tf.nn.dropout(h_pool, keep_prob)
        h_pool_flat = tf.reshape(h_pool_drop, [-1, SHAPE_convlast])
        h_rnn = tf.nn.relu(tf.matmul(h_pool_flat,self.W_pre_rnn ) + self.b_pre_rnn)
        h_rnn = tf.reshape(h_rnn, (self.batch_size, int(self.encoder_time_size*2/self.frame_rate), self.rnn_hid_dim))
        # enc-rnn
        outputs, end_states = tf.contrib.rnn.static_rnn(enc_cell, tf.unstack(tf.transpose(h_rnn, [1, 0, 2])), dtype=tf.float32)
        output = tf.reshape(outputs[-1], [-1, self.rnn_hid_dim])

        output_drop = output
        for layer_ind in xrange(len(self.fc_layers) - 2):
            output_drop = tf.nn.relu(tf.matmul(output_drop, W_fc[layer_ind]) + b_fc[layer_ind])
            output_drop = tf.nn.dropout(output_drop, keep_prob)

        # if batch norm lstm layer, add orthogonal weights to last fc layer
        if self.cell_type != 'BNLSTMCell':
            W_fc[-1] = tf.get_variable('w_rnn', [self.fc_layers[-2], self.fc_layers[-1]], initializer=orthogonal_initializer())
        y = tf.matmul(output_drop, W_fc[-1]) + b_fc[-1]

        self.x = x
        self.keep_prob = keep_prob
        self.logits = y

    def input(self, x, keep_prob=None, training=False):
        ret_dict = {}
        if keep_prob == None: #default, 'training'
            keep_prob = self.value_keep_prob
        if self.bn:
            ret_dict[self.training] = training
        ret_dict[self.x] = x
        ret_dict[self.keep_prob] = keep_prob
        return ret_dict

    def output(self):
        return self.logits
