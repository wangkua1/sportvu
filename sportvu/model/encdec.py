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
        self.dec_rnn_hid_dim = config['dec_rnn_hid_dim']
        self.encoder_input_shape = config['encoder_input_shape']

    def build(self):
        # placeholders
        tf_dec_input = tf.placeholder(tf.float32, [self.batch_size, self.decoder_time_size, 2])
        keep_prob = tf.placeholder(tf.float32)
        self.teacher_forcing_stop = tf.placeholder(tf.int32)
        
        # init weights/bias
        # dec pre-rnn
        self.W_dec_inp_hid = weight_variable([2, self.dec_rnn_hid_dim])
        self.b_dec_inp_hid = bias_variable([self.dec_rnn_hid_dim])
        ## rnn 
        # Dec
        dec_cell = tf.contrib.rnn.BasicLSTMCell(self.dec_rnn_hid_dim, state_is_tuple=True)
        # dec post-rnn
        self.W_dec_out_hid = weight_variable([self.dec_rnn_hid_dim, 2])
        # self.b_dec_out_hid = bias_variable([2]) ### probably don't need output bias
        
        ## Build Graph
        # build decoder
        dec_outputs = []
        initial_state = dec_cell.zero_state(self.batch_size, tf.float32) ## TODO: use encoder states
        state = initial_state
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
        
        
        # self.tf_enc_input = tf_enc_input
        self.tf_dec_input = tf_dec_input
        # self.keep_prob = keep_prob
        self.outputs = tf.transpose(dec_outputs, [1,0,2]) # -> (BATCH, TIME, 2)

    def input(self, dec_input, teacher_forcing_stop = None):
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
        return ret_dict

    def output(self):
        return self.outputs


if __name__ == '__main__':


    import yaml
    optimize_loss = tf.contrib.layers.optimize_loss

    f_model_config = 'config/dec-single-frame.yaml'
    model_config = yaml.load(open(f_model_config, 'rb'))['model_config']
    model_config['keep_prob'] = 1
    ## Fake Data
    # enc_input = np.random.rand(model_config['batch_size'], 4, model_config['encoder_time_size'], model_config['d1'], model_config['d2'] )
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
    feed_dict = net.input(dec_input)
    feed_dict[learning_rate] = 1e-4
    feed_dict[y_] = dec_output
    for train_step_ind in xrange(10):
        l = sess.run(train_step, feed_dict=feed_dict)
        print (l)
    print ('.........')
    feed_dict = net.input(dec_input, teacher_forcing_stop=1)
    feed_dict[learning_rate] = 1e-4
    feed_dict[y_] = dec_output
    for train_step_ind in xrange(10):
        l = sess.run(train_step, feed_dict=feed_dict)
        print (l)


    ### test teacher-forcing
    init_dec = np.random.rand(model_config['batch_size'], 2)
    print ('.........')
    ## following results should be the different
    for _ in xrange(5):
        dec_input = np.random.rand(model_config['batch_size'],  model_config['decoder_time_size'], 2 )
        dec_input[:,0] = init_dec
        feed_dict = net.input(dec_input, teacher_forcing_stop=None)
        feed_dict[y_] = dec_output
        l = sess.run(euclid_loss, feed_dict= feed_dict)
        print (l)
    print ('.........')
    ## following results should be the same (prediction mode)
    for _ in xrange(5):
        dec_input = np.random.rand(model_config['batch_size'],  model_config['decoder_time_size'], 2 )
        dec_input[:,0] = init_dec
        feed_dict = net.input(dec_input, teacher_forcing_stop=1)
        feed_dict[y_] = dec_output
        l = sess.run(euclid_loss, feed_dict= feed_dict)
        print (l)

