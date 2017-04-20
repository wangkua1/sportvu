from __future__ import division
import tensorflow as tf
import numpy as np
from utils import * 

class Seq2Seq:
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
        self.channel_size = 4
        self.encoder_time_size = config['encoder_time_size']
        self.decoder_time_size = config['decoder_time_size']
        self.rnn_hid_dim = config['rnn_hid_dim']
        self.batch_size = config['batch_size']
        self.teacher_forcing = False

    def build(self):
        # placeholders
        tf_dec_input = tf.placeholder(tf.float32, [self.batch_size, self.channel_size, self.decoder_time_size, self.d1, self.d2])
        keep_prob = tf.placeholder(tf.float32)
        tf_dec_target_xy = tf.placeholder(tf.float32, [self.batch_size, self.decoder_time_size, 2])
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
        cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_hid_dim, state_is_tuple=True)
        # fc (post-rnn)
        W_fc = []
        b_fc = []
        self.fc_layers.insert(0, self.rnn_hid_dim)
        for layer_ind in xrange(len(self.fc_layers) - 1):
            W_fc.append(weight_variable(
                [self.fc_layers[layer_ind], self.fc_layers[layer_ind + 1]]))
            b_fc.append(bias_variable([self.fc_layers[layer_ind + 1]]))
        
        # build decoder
        dec_outputs = []
        initial_state = cell.zero_state(self.batch_size, tf.float32) ## TODO: use encoder states
        state = initial_state
        with tf.variable_scope("dec_rnn") as scope:
            for rnn_step_ind, input_ in enumerate(tf.unstack(tf.transpose(tf_dec_input, [2,0,3,4,1]))):
                if rnn_step_ind > 0:
                    scope.reuse_variables()
                    if self.teacher_forcing: ## feed in prediction
                        ## output (BATCH, 2)
                        if rnn_step_ind == 1:
                            ref = tf_dec_target_xy[:, 0]
                        else:
                            ref = abs_output
                        abs_output = output + ref
                        indices = tf.add(tf.scalar_mul(self.d2, abs_output[:,0]),abs_output[:,1])
                        output = tf.one_hot(tf.cast(indices, tf.int32), self.d1*self.d2)
                        output = tf.reshape(output, (self.batch_size, self.d1, self.d2))
                        input_[:,:,:,0] = output
                    else:
                        pass
                else: ## first step, always feed-in gt
                    pass
                # conv
                h_pool_drop = input_
                for layer_ind in xrange(len(self.conv_layers)):
                    h_conv = tf.nn.relu(
                        conv2d(h_pool_drop, W_conv[layer_ind]) + b_conv[layer_ind])
                    h_pool = max_pool_2x2(h_conv)
                    h_pool_drop = tf.nn.dropout(h_pool, keep_prob)
                h_pool_flat = tf.reshape(h_pool_drop, [-1, SHAPE_convlast])
                h_rnn = tf.nn.relu(tf.matmul(h_pool_flat,self.W_pre_rnn ) + self.b_pre_rnn)
                ## RNN cell
                h_rnn, state = cell(h_rnn, state)
                # fc output
                h_fc_drop = h_rnn
                for layer_ind in xrange(len(self.fc_layers) - 2):
                    h_fc = tf.nn.relu(tf.matmul(h_fc_drop, W_fc[
                                      layer_ind]) + b_fc[layer_ind])
                    h_fc_drop = tf.nn.dropout(h_fc, keep_prob)

                output = tf.matmul(h_fc_drop, W_fc[-1]) + b_fc[-1] 
                dec_outputs.append(output)
        
        

        self.tf_dec_input = tf_dec_input
        self.keep_prob = keep_prob
        self.outputs = tf.transpose(dec_outputs, [1,0,2]) # -> (BATCH, TIME, 2)

    def input(self, dec_input, keep_prob=None):
        if keep_prob == None: #default, 'training'
            keep_prob = self.value_keep_prob
        ret_dict = {}
        ret_dict[self.tf_dec_input] = dec_input
        ret_dict[self.keep_prob] = keep_prob
        return ret_dict

    def output(self):
        return self.outputs


if __name__ == '__main__':
    # ### figuring out how to use tf.one_hot
    # Y_RANGE = 3
    # X_RANGE = 2
    # batch_size = 5
    # previous_pred = tf.placeholder(tf.float32, [batch_size, 2])
    # indices = tf.add(tf.scalar_mul(X_RANGE, previous_pred[:,0]),previous_pred[:,1])
    # output = tf.one_hot(tf.cast(indices, tf.int32), Y_RANGE*X_RANGE)
    # output = tf.reshape(output, (batch_size,Y_RANGE,X_RANGE))
    # sess = tf.InteractiveSession()
    # nout = sess.run(output, feed_dict={previous_pred:np.array([[2.1, 0]]).repeat(5,0)})

    # # raise

    import yaml
    optimize_loss = tf.contrib.layers.optimize_loss

    f_model_config = 'config/seq2seq-1.yaml'
    model_config = yaml.load(open(f_model_config, 'rb'))['model_config']
    model_config['keep_prob'] = 1
    ## Fake Data
    dec_input = np.random.rand(model_config['batch_size'], 4, model_config['decoder_time_size'], model_config['d1'], model_config['d2'] )
    dec_output = np.random.rand(model_config['batch_size'], model_config['decoder_time_size'], 2)
    dec_target_sequence = np.random.rand(model_config['batch_size'], model_config['decoder_time_size'], 2) 
    ## Build Model    
    net = Seq2Seq(model_config)
    net.build()
    ## Build Loss
    y_ = tf.placeholder(tf.float32, [model_config['batch_size'], model_config['decoder_time_size'], 2])
    euclid_loss = tf.reduce_mean(tf.pow(net.output() - y_, 2))
    ## Build Learning 
    learning_rate = tf.placeholder(tf.float32, [])
    global_step = tf.Variable(0)
    train_step = optimize_loss(euclid_loss, global_step, learning_rate, 
                optimizer=lambda lr: tf.train.AdamOptimizer(lr))

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    feed_dict = net.input(dec_input)
    feed_dict[learning_rate] = 1e-4
    feed_dict[y_] = dec_output
    for train_step_ind in xrange(10):
        l = sess.run(train_step, feed_dict=feed_dict)
        print (l)
    net.teacher_forcing = True
    print ('.........')
    for train_step_ind in xrange(10):
        l = sess.run(train_step, feed_dict=feed_dict)
        print (l)
    





    # learning_rate = tf.placeholder(tf.float32, [])

    # cell = tf.contrib.rnn.BasicLSTMCell(n_input, state_is_tuple=True)
    # W_out = weight_variable([n_input,1])
    # initial_state = cell.zero_state(batch_size, tf.float32)
    # # rnn_outputs, rnn_states = tf.contrib.rnn.static_rnn(cell, tf.unstack(tf.transpose(x, [1,0,2])), dtype=tf.float32)
    # rnn_outputs = []
    # state = initial_state
    # with tf.variable_scope("myrnn") as scope:
    #     for rnn_step, input_ in enumerate(tf.unstack(tf.transpose(x, [1,0,2]))):
    #         if rnn_step > 0:
    #             scope.reuse_variables()
    #             ii = tf.reshape(reduced_output, [batch_size])
    #             output = tf.one_hot(tf.cast(ii, tf.int32),n_input)
    #             # output = tf.zeros([batch_size, n_input], dtype=tf.float32)
    #             # output[tf.range(0,batch_size,1), reduced_output] = 1.
    #         else:
    #             output = input_
    #         output, state = cell(output, state)
    #         ### output layer
    #         reduced_output =tf.scalar_mul( n_input,tf.nn.sigmoid(tf.matmul(output, W_out)))
    #         rnn_outputs.append(output)
    
    # rnn_states = state
    # last_output = rnn_outputs[-1]
    # euclid_loss = tf.reduce_mean(tf.pow(last_output - y, 2))
    # global_step = tf.Variable(0)
    # train_step = optimize_loss(euclid_loss, global_step, learning_rate, 
    #             optimizer=lambda lr: tf.train.AdamOptimizer(lr))

    # sess = tf.InteractiveSession()
    # tf.global_variables_initializer().run()
    # outputs, states, loss = sess.run([rnn_outputs, rnn_states, euclid_loss], 
    #                         feed_dict={x:np.random.rand(batch_size, n_steps, n_input),
    #                                    y:np.random.rand(batch_size, n_input)})
    # y_inp = np.random.rand(batch_size, n_input)
    # x_inp = np.random.rand(batch_size, n_steps, n_input)
    # for train_step_ind in xrange(10):
    #     l = sess.run(train_step,  
    #                         feed_dict={x:x_inp,
    #                                    y:y_inp,
    #                                    learning_rate:1e-4})
    #     print (l)
   
