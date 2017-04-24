from __future__ import division
import tensorflow as tf
import numpy as np
from utils import * 

class EncDec:
    """
    Encoder-Decoder Model (for single sequence future prediction problem)
        Encoder could be arbitrarily complex
            passes the final state through non-linearity to initlize decoder state
        Decoder is alwasy very simple input and output are the same format, just
                2-dimensional delta (x,y)
    """

    def __init__(self, config):
        self.config = config
        self.batch_size = config['batch_size']    
        self.decoder_time_size = config['decoder_time_size']
        self.enc_rnn_hid_dim = config['enc_rnn_hid_dim']
        self.dec_rnn_hid_dim = config['dec_rnn_hid_dim']
        assert (self.enc_rnn_hid_dim == self.dec_rnn_hid_dim) ## right now doesn't support rnns of diff size, feels it might be harder to optimize
        self.encoder_input_shape = config['encoder_input_shape']
        if self.encoder_input_shape is not None:
            self.fc_layers = config['fc_layers']
            if len(self.encoder_input_shape) == 5:
                self.conv_layers = config['conv_layers']
            self.keep_prob_value = config['keep_prob']

    def build(self):
        # placeholders
        tf_dec_input = tf.placeholder(tf.float32, [self.batch_size, self.decoder_time_size, 2])
        keep_prob = tf.placeholder(tf.float32)
        self.teacher_forcing_stop = tf.placeholder(tf.int32)
        tf_enc_input = tf.placeholder(tf.float32, self.encoder_input_shape) ## either (N, T, D), or (N, C, T, Y, X)
        # init weights/bias
        # [enc] pre-rnn Conv
        if self.encoder_input_shape is not None and len(self.encoder_input_shape) == 5:
            self.batch_size = self.encoder_input_shape[0]
            self.channel_size = self.encoder_input_shape[1]
            self.encoder_time_size = self.encoder_input_shape[2]
            self.d1 = self.encoder_input_shape[3]
            self.d2 = self.encoder_input_shape[4]
            W_conv = []
            b_conv = []
            for layer_ind in xrange(len(self.conv_layers)):
                W_conv.append(weight_variable(self.conv_layers[layer_ind]))
                b_conv.append(bias_variable([self.conv_layers[layer_ind][-1]]))
           
            SHAPE_convlast = int(np.ceil(self.d1 / (2**len(self.conv_layers))) *
                                 np.ceil(self.d2 / (2**len(self.conv_layers))) *
                                 self.conv_layers[-1][-1])
        if self.encoder_input_shape is not None:
            # [enc] pre-rnn FC
            W_fc = []
            b_fc = []
            # first fc shape
            if len(self.encoder_input_shape) == 5:
                shape_zero = SHAPE_convlast
            else:
                shape_zero = self.encoder_input_shape[-1]
            self.fc_layers.insert(0, shape_zero)
            for layer_ind in xrange(len(self.fc_layers) - 1):
                W_fc.append(weight_variable(
                    [self.fc_layers[layer_ind], self.fc_layers[layer_ind + 1]]))
                b_fc.append(bias_variable([self.fc_layers[layer_ind + 1]]))
            # [enc] rnn
            enc_cell = tf.contrib.rnn.BasicLSTMCell(self.enc_rnn_hid_dim, state_is_tuple=True)
        # [glue] 2 linear weights taking enc states to decoder
        tf_glue_1 = tf.Variable(tf.eye(self.dec_rnn_hid_dim))
        tf_glue_2 = tf.Variable(tf.eye(self.dec_rnn_hid_dim))
        # [dec] pre-rnn
        self.W_dec_inp_hid = weight_variable([2, self.dec_rnn_hid_dim])
        self.b_dec_inp_hid = bias_variable([self.dec_rnn_hid_dim])
        # [dec] rnn 
        dec_cell = tf.contrib.rnn.BasicLSTMCell(self.dec_rnn_hid_dim, state_is_tuple=True)
        # [dec] post-rnn
        self.W_dec_out_hid = weight_variable([self.dec_rnn_hid_dim, 2])
        # self.b_dec_out_hid = bias_variable([2]) ### probably don't need output bias
        
        ## Build Graph
        # build eccoder
        if self.encoder_input_shape is not None:
            if len(self.encoder_input_shape) == 5:
                # tf_enc_input (B, C, T, Y, X)
                
                #
                tf_r_enc_input = tf.transpose( tf_enc_input, (0,2,3,4,1)) # (B,T,Y,X,C)
                tf_r_enc_input = tf.reshape(tf_r_enc_input, # (B*T, Y,X,C)
                                    (self.batch_size*self.encoder_time_size, self.d1,self.d2,self.channel_size))
                # conv
                h_pool_drop = tf_r_enc_input
                for layer_ind in xrange(len(self.conv_layers)):
                    h_conv = tf.nn.relu(
                        conv2d(h_pool_drop, W_conv[layer_ind]) + b_conv[layer_ind])
                    h_pool = max_pool_2x2(h_conv)
                    h_pool_drop = tf.nn.dropout(h_pool, keep_prob)
                h_fc_drop = tf.reshape(h_pool_drop, [-1, SHAPE_convlast])
                for layer_ind in xrange(len(self.fc_layers) - 1):
                    h_fc = tf.nn.relu(tf.matmul(h_fc_drop, W_fc[
                                      layer_ind]) + b_fc[layer_ind])
                    h_fc_drop = tf.nn.dropout(h_fc, keep_prob)
                h_rnn = tf.reshape(h_fc_drop, (self.batch_size, self.encoder_time_size, self.enc_rnn_hid_dim))
            elif len(self.encoder_input_shape)== 3:
                # tf_enc_input (B, T, D)
                self.batch_size = self.encoder_input_shape[0]
                self.encoder_time_size = self.encoder_input_shape[1]
                self.d = self.encoder_input_shape[2]
                #
                tf_r_enc_input = tf.reshape(tf_enc_input, # (B*T,D)
                                    (self.batch_size*self.encoder_time_size, self.d))
                h_fc_drop = tf_r_enc_input
                for layer_ind in xrange(len(self.fc_layers) - 1):
                    h_fc = tf.nn.relu(tf.matmul(h_fc_drop, W_fc[
                                      layer_ind]) + b_fc[layer_ind])
                    h_fc_drop = tf.nn.dropout(h_fc, keep_prob)
                h_rnn = tf.reshape(h_fc_drop, (self.batch_size, self.encoder_time_size, self.enc_rnn_hid_dim))
            # enc-rnn
            _, enc_states = tf.contrib.rnn.static_rnn(enc_cell, tf.unstack(tf.transpose(h_rnn, [1,0,2])), dtype=tf.float32)
        ##

        # build decoder
        dec_outputs = []
        if self.encoder_input_shape is not None:
            s1 = tf.matmul(enc_states[0],tf_glue_1)
            s2 = tf.matmul(enc_states[1],tf_glue_2)
            state = (s1,s2)
        else:
            state = dec_cell.zero_state(self.batch_size, tf.float32) 
        with tf.variable_scope("dec_rnn") as scope:
            for rnn_step_ind, input_ in enumerate(tf.unstack(tf.transpose(tf_dec_input, [1,0,2]))):
                if rnn_step_ind > 0:
                    scope.reuse_variables()
                    ## output (BATCH, 2)
                    ## select
                    tf_step_ind = tf.Variable(rnn_step_ind)
                    input_ = tf.where(tf.greater_equal(tf_step_ind, self.teacher_forcing_stop),  output, input_)
                else: ## first step, always feed-in gt
                    pass
                h_fc = tf.nn.relu(tf.matmul(input_, self.W_dec_inp_hid) + self.b_dec_inp_hid)
                h_rnn = h_fc
                ## RNN cell
                h_rnn, state = dec_cell(h_rnn, state)
                # fc output
                output = tf.matmul(h_rnn, self.W_dec_out_hid) 
                dec_outputs.append(output)
        
        
        self.tf_enc_input = tf_enc_input
        self.tf_dec_input = tf_dec_input
        self.keep_prob = keep_prob
        self.outputs = tf.transpose(dec_outputs, [1,0,2]) # -> (BATCH, TIME, 2)

    def input(self, dec_input, teacher_forcing_stop = None, enc_input=None, enc_keep_prob=None):
        # if keep_prob == None: #default, 'training'
        #     keep_prob = self.value_keep_prob
        ret_dict = {}
        # ret_dict[self.tf_enc_input] = enc_input
        ret_dict[self.tf_dec_input] = dec_input
        if teacher_forcing_stop == None: # default, always teacher-force
            ret_dict[self.teacher_forcing_stop] = int(self.decoder_time_size) 
        else:
            assert (teacher_forcing_stop >= 1) # has to at least feed in the first frame
            ret_dict[self.teacher_forcing_stop] = int(teacher_forcing_stop)
        if enc_input is not None:
            ret_dict[self.tf_enc_input] = enc_input
            if enc_keep_prob is None:
                enc_keep_prob = self.keep_prob_value
            ret_dict[self.keep_prob] = enc_keep_prob
        return ret_dict

    def output(self):
        return self.outputs


if __name__ == '__main__':


    import yaml
    optimize_loss = tf.contrib.layers.optimize_loss

    f_model_config = 'config/ed-full-3d.yaml'
    model_config = yaml.load(open(f_model_config, 'rb'))['model_config']
    model_config['keep_prob'] = 1
    ## Fake Data
    if model_config['encoder_input_shape'] is not None:
        enc_input = np.random.rand(*model_config['encoder_input_shape'])
        keep_prob = model_config['keep_prob']
    dec_input = np.random.rand(model_config['batch_size'],  model_config['decoder_time_size'], 2 )
    dec_output = np.random.rand(model_config['batch_size'], model_config['decoder_time_size'], 2)
    ## Build Model    
    net = EncDec(model_config)
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
    if model_config['encoder_input_shape'] is not None:
        feed_dict = net.input(dec_input,teacher_forcing_stop = None, enc_input=enc_input, enc_keep_prob=keep_prob)
    else:
        feed_dict = net.input(dec_input)
    feed_dict[learning_rate] = 1e-4
    feed_dict[y_] = dec_output
   
    for train_step_ind in xrange(10):
        l = sess.run(train_step, feed_dict=feed_dict)
        print (l)
    print ('.........')


    # ### test teacher-forcing [PASSED]
    # init_dec = np.random.rand(model_config['batch_size'], 2)
    # print ('.........')
    # ## following results should be the different
    # for _ in xrange(5):
    #     dec_input = np.random.rand(model_config['batch_size'],  model_config['decoder_time_size'], 2 )
    #     dec_input[:,0] = init_dec
    #     feed_dict = net.input(dec_input, teacher_forcing_stop=None)
    #     feed_dict[y_] = dec_output
    #     l = sess.run(euclid_loss, feed_dict= feed_dict)
    #     print (l)
    # print ('.........')
    # ## following results should be the same (prediction mode)
    # for _ in xrange(5):
    #     dec_input = np.random.rand(model_config['batch_size'],  model_config['decoder_time_size'], 2 )
    #     dec_input[:,0] = init_dec
    #     feed_dict = net.input(dec_input, teacher_forcing_stop=1)
    #     feed_dict[y_] = dec_output
    #     l = sess.run(euclid_loss, feed_dict= feed_dict)
    #     print (l)

