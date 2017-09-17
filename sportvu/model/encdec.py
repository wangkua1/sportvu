from __future__ import division
import tensorflow as tf
from tensorflow.python.ops.distributions.util import fill_lower_triangular
import numpy as np
from utils import * 




class EncDec(object):
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
        self.dec_input_dim = config['dec_input_dim']
        self.dec_output_dim = config['dec_output_dim']
        assert (self.enc_rnn_hid_dim == self.dec_rnn_hid_dim) ## right now doesn't support rnns of diff size, feels it might be harder to optimize
        self.encoder_input_shape = config['encoder_input_shape']
        if self.encoder_input_shape is not None:
            self.fc_layers = config['fc_layers']
            if len(self.encoder_input_shape) == 5:
                self.conv_layers = config['conv_layers']
            self.keep_prob_value = config['keep_prob']
        if "decoder_init_noise" in config: #stochasticity
            self.decoder_init_noise = config['decoder_init_noise']
            self.decoder_noise_level  = config['noise_level']
        else:
            self.decoder_init_noise = None
            self.decoder_noise_level = None
        if "decoder_input_keep_prob" in config:
            self.decoder_input_keep_prob = config['decoder_input_keep_prob']
        else:
            self.decoder_input_keep_prob = None
        self.output_format = config['output_format']
        self.tf_start_frame = tf.placeholder(tf.float32, [self.batch_size, self.dec_input_dim])

        ### Initialize TF variables

        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        if self.decoder_init_noise is not None:
            self.pl_decoder_noise_level = tf.placeholder(tf.float32, [], name='dec_noise_lvl')
        if self.decoder_input_keep_prob is not None:
            self.pl_decoder_input_keep_prob = tf.placeholder(tf.float32, [], name='dec_input_keep_prob')
        else:
            self.pl_decoder_input_keep_prob = tf.constant(1.)

        self.teacher_forcing_stop = tf.placeholder(tf.int32, name='tf_stop')
        tf_enc_input = tf.placeholder(tf.float32, self.encoder_input_shape, name='enc_input') ## either (N, T, D), or (N, C, T, Y, X)
        
        self.tf_enc_input = tf_enc_input
        # self.tf_dec_input = tf_dec_input
        self.keep_prob = keep_prob

        # init weights/bias
        # [enc] pre-rnn Conv
        if self.encoder_input_shape is not None and len(self.encoder_input_shape) == 5:
            self.batch_size = self.encoder_input_shape[0]
            self.channel_size = self.encoder_input_shape[1]
            self.encoder_time_size = self.encoder_input_shape[2]
            self.d1 = self.encoder_input_shape[3]
            self.d2 = self.encoder_input_shape[4]
            self.W_conv = []
            self.b_conv = []
            for layer_ind in xrange(len(self.conv_layers)):
                self.W_conv.append(weight_variable(self.conv_layers[layer_ind]))
                self.b_conv.append(bias_variable([self.conv_layers[layer_ind][-1]]))
           
            SHAPE_convlast = int(np.ceil(self.d1 / (2**len(self.conv_layers))) *
                                 np.ceil(self.d2 / (2**len(self.conv_layers))) *
                                 self.conv_layers[-1][-1])
        if self.encoder_input_shape is not None:
            # [enc] pre-rnn FC
            self.W_fc = []
            self.b_fc = []
            # first fc shape
            if len(self.encoder_input_shape) == 5:
                shape_zero = SHAPE_convlast
            else:
                shape_zero = self.encoder_input_shape[-1]
            self.fc_layers.insert(0, shape_zero)
            for layer_ind in xrange(len(self.fc_layers) - 1):
                self.W_fc.append(weight_variable(
                    [self.fc_layers[layer_ind], self.fc_layers[layer_ind + 1]]))
                self.b_fc.append(bias_variable([self.fc_layers[layer_ind + 1]]))
            # [enc] rnn
            self.enc_cell = tf.contrib.rnn.BasicLSTMCell(self.enc_rnn_hid_dim, state_is_tuple=True)
        # [glue] 2 linear weights taking enc states to decoder
        self.tf_glue_1 = tf.Variable(tf.eye(self.dec_rnn_hid_dim))
        self.tf_glue_2 = tf.Variable(tf.eye(self.dec_rnn_hid_dim))
        # [dec] pre-rnn
        self.W_dec_inp_hid = weight_variable([self.dec_input_dim, self.dec_rnn_hid_dim])
        self.b_dec_inp_hid = bias_variable([self.dec_rnn_hid_dim])
        # [dec] rnn 
        self.dec_cell = tf.contrib.rnn.BasicLSTMCell(self.dec_rnn_hid_dim, state_is_tuple=True)
        # [dec] post-rnn
        self.W_dec_out_hid = weight_variable([self.dec_rnn_hid_dim, self.dec_output_dim])
        # self.b_dec_out_hid = bias_variable([2]) ### probably don't need output bias
        


    def build(self, tf_input=None):
        tf_enc_input = self.tf_enc_input
        keep_prob = self.keep_prob
        # placeholders
        if tf_input is None:
            tf_dec_input = tf.placeholder(tf.float32, [self.batch_size, self.decoder_time_size, self.dec_input_dim])
        else:
            tf_dec_input = tf_input
        self.tf_dec_input = tf_dec_input
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
                        conv2d(h_pool_drop, self.W_conv[layer_ind]) + self.b_conv[layer_ind])
                    h_pool = max_pool_2x2(h_conv)
                    h_pool_drop = tf.nn.dropout(h_pool, keep_prob)
                h_fc_drop = tf.reshape(h_pool_drop, [-1, SHAPE_convlast])
                for layer_ind in xrange(len(self.fc_layers) - 1):
                    h_fc = tf.nn.relu(tf.matmul(h_fc_drop, self.W_fc[
                                      layer_ind]) + self.b_fc[layer_ind])
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
                    h_fc = tf.nn.relu(tf.matmul(h_fc_drop, self.W_fc[
                                      layer_ind]) + self.b_fc[layer_ind])
                    h_fc_drop = tf.nn.dropout(h_fc, keep_prob)
                h_rnn = tf.reshape(h_fc_drop, (self.batch_size, self.encoder_time_size, self.enc_rnn_hid_dim))
            # enc-rnn
            _, enc_states = tf.contrib.rnn.static_rnn(self.enc_cell, tf.unstack(tf.transpose(h_rnn, [1,0,2])), dtype=tf.float32)
        ##

        # build decoder
        dec_outputs = []
        sampled_outputs = []
        if self.encoder_input_shape is not None:
            s1 = tf.matmul(enc_states[0],self.tf_glue_1)
            s2 = tf.matmul(enc_states[1],self.tf_glue_2)
            state = (s1,s2)
        else:
            state = self.dec_cell.zero_state(self.batch_size, tf.float32) 
        # stochasticity
        if self.decoder_init_noise is not None:
            if self.decoder_init_noise == 'gaussian':
                s0=gaussian_noise_layer(state[0], self.pl_decoder_noise_level)
                s1=gaussian_noise_layer(state[1], self.pl_decoder_noise_level)
                state = (s0,s1)
            elif self.decoder_init_noise == 'dropout':
                s0 = tf.nn.dropout(state[0], tf.constant(1.) - self.pl_decoder_noise_level)
                s1 = tf.nn.dropout(state[1], tf.constant(1.) - self.pl_decoder_noise_level)
                state = (s0, s1)
            else:
                raise NotImplementedError()
        with tf.variable_scope("dec_rnn") as scope:
            for rnn_step_ind, input_ in enumerate(tf.unstack(tf.transpose(tf_dec_input, [1,0,2]))):
                if rnn_step_ind > 0:
                    scope.reuse_variables()
                    ## output (BATCH, 2)
                    ## select
                    tf_step_ind = tf.Variable(rnn_step_ind)
                    input_ = tf.where(tf.greater_equal(tf_step_ind, self.teacher_forcing_stop), sampled_output , input_)
                    input_ = tf.nn.dropout(input_, self.pl_decoder_input_keep_prob)
                else: ## first step, always feed-in gt
                    pass
                h_fc = tf.nn.relu(tf.matmul(input_, self.W_dec_inp_hid) + self.b_dec_inp_hid)
                h_rnn = h_fc
                ## RNN cell
                h_rnn, state = self.dec_cell(h_rnn, state)
                # fc output
                output = tf.matmul(h_rnn, self.W_dec_out_hid) 
                dec_outputs.append(output)
                sampled_output = self.sample_timestep(output)
                sampled_outputs.append(sampled_output)
        
        
        self.outputs = tf.transpose(dec_outputs, [1,0,2]) # -> (BATCH, TIME, 2)
        self.sampled_outputs = tf.transpose(sampled_outputs, [1,0,2]) # -> (BATCH, TIME, 2)
        return self.outputs

    def input(self, dec_input, teacher_forcing_stop = None, 
                enc_input=None, enc_keep_prob=None,decoder_noise_level=None, decoder_input_keep_prob=None, ret_dict=None):
        # if keep_prob == None: #default, 'training'
        #     keep_prob = self.value_keep_prob
        if ret_dict is None:
            ret_dict = {}
        # ret_dict[self.tf_enc_input] = enc_input
        if dec_input is not None:
            ret_dict[self.tf_dec_input] = dec_input
        if teacher_forcing_stop == None: # default, always teacher-force
            ret_dict[self.teacher_forcing_stop] = int(self.decoder_time_size) 
        else:
            assert (teacher_forcing_stop >= 1) # has to at least feed in the first frame
            ret_dict[self.teacher_forcing_stop] = int(teacher_forcing_stop)
        if self.encoder_input_shape is not None:
            ret_dict[self.tf_enc_input] = enc_input
            if enc_keep_prob is None:
                enc_keep_prob = self.keep_prob_value
            ret_dict[self.keep_prob] = enc_keep_prob
        if self.decoder_init_noise is not None:
            if decoder_noise_level is None:
                ret_dict[self.pl_decoder_noise_level] = self.decoder_noise_level
            else:
                ret_dict[self.pl_decoder_noise_level] = decoder_noise_level
        if self.decoder_input_keep_prob is not None:
            if decoder_input_keep_prob is None:
                ret_dict[self.pl_decoder_input_keep_prob] = self.decoder_input_keep_prob
            else:
                ret_dict[self.pl_decoder_input_keep_prob] = decoder_input_keep_prob
        return ret_dict

    def output(self):
        return self.outputs

    def sample(self):
        return self.sampled_outputs

    def aux_feed_dict(self, aux, feed_dict):
        feed_dict[self.tf_start_frame] = aux['start_frame']

    def sample_trajectory(self):
        if self.output_format == 'location':
            return self.sample()
        elif self.output_format == 'velocity':
            vels = self.sample() # (N, T, D)
            traj = [self.tf_start_frame]
            for time_idx in xrange(self.decoder_time_size):
                traj.append(vels[:,time_idx] +traj[-1])
            return tf.transpose(traj[1:], [1,0,2])
        else:
            raise Exception('specify output_format in model config, [{}]'.format(self.output_format))

class Probabilistic(EncDec):
    """docstring for Location"""
    def __init__(self, arg):
        super(Probabilistic, self).__init__(arg)
        if 'sample' in arg and arg['sample'] is not None:
            self.do_sample = True
            self.sample_scheme = arg['sample'] # right now just 'gauss_diag','gauss_full', 'gauss_mean'
        else:
            self.do_sample = False

    def sample_timestep(self, curr_output, eps=1e-5):
        """
        curr_output is a tf variable holding the network output at the current time step
        sample_timestep() should return a sample in the same shape as decoder_input (self.dec_input_dim])
        """
        if not self.do_sample: ## deterministic
            return curr_output

        if self.sample_scheme == 'gauss_mean':
            return curr_output[..., :self.dec_input_dim]
        elif self.sample_scheme == 'gauss_diag':
            mean, pre_var = tf.split(curr_output, 2, axis=1)
            var = tf.nn.softplus(pre_var) + eps
            normal = tf.random_normal(mean.shape, 0, 1,
                                   dtype=tf.float32)
            return tf.add(mean, tf.multiply(tf.sqrt(var), normal))
        elif self.sample_scheme == 'gauss_full':
            tot_dim = curr_output.get_shape()[-1].value
            pred_dim = int((-3 + np.sqrt(9 + 8 * tot_dim)) / 2)
            assert(pred_dim+pred_dim * (pred_dim + 1) / 2 == tot_dim)
            mean, R = tf.split(curr_output, [int(pred_dim), int(pred_dim * (pred_dim + 1) / 2)],
                               axis=1)  # Sigma_inv = RR^T Cholasky decomp, Sigma = R^(-T)R^(-1)
            R_trans = tf.transpose(fill_lower_triangular(tf.nn.softplus(R) + eps),[0, 2, 1])

            normal_shape = [s.value for s in mean.shape]
            normal_shape.append(1)
            normal = tf.random_normal(normal_shape, 0, 1,
                                   dtype=tf.float32)
            return tf.add(mean, tf.reshape(tf.matrix_triangular_solve(R_trans, normal), mean.shape)) #R^-T*normal has Sigma R^(-T)R^(-1)

        else:
            raise Exception('unknown sampling scheme')




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

