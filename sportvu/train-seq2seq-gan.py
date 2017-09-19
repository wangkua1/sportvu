"""train-seq2seq-gan.py

Usage:
    train-seq2seq-gan.py <fold_index> <f_data_config> <f_model_config> <loss> [--prefix <p>]
    train-seq2seq-gan.py --test <fold_index> <f_data_config> <f_model_config>

Arguments:
    <f_data_config>  example ''data/config/train_rev0.yaml''
    <f_model_config> example 'model/config/conv2d-3layers.yaml'

Example:
    python train.py 0 data/config/train_rev0.yaml model/config/conv2d-3layers.yaml
    python train.py 0 data/config/train_rev0_vid.yaml model/config/conv3d-1.yaml
Options:
    --negative_fraction_hard=<percent> [default: 0]
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
# model
import tensorflow as tf
optimize_loss = tf.contrib.layers.optimize_loss
import sys
import os
from sportvu.model.seq2seq import Seq2Seq
from sportvu.model.encdec import EncDec, Probabilistic
# data
from sportvu.data.dataset import BaseDataset
from sportvu.data.extractor import Seq2SeqExtractor, EncDecExtractor
from sportvu.data.loader import Seq2SeqLoader
from sportvu.data.utils import wrapper_concatenated_last_dim, scale_last_dim
# loss
from loss import RMSEPerPlayer, DiagGaussNLL, FullGaussNLL
from tqdm import tqdm
from docopt import docopt
import yaml
import gc
from utils import truncated_mean, experpolate_position,dist_trajectory
from vis_utils import make_sequence_prediction_image
import cPickle as pkl
# import matplotlib.pylab as plt
# plt.ioff()
# fig = plt.figure()


### [Directory bookkeeping]
# checkpoints
if not os.path.exists('./gan-saves'):
    os.mkdir('./gan-saves')
# tensorboard
if not os.path.exists('./gan-logs'):
    os.mkdir('./gan-logs')
### 



CRITIC_ITERS = 5 
MODE = 'wgan'

init_lr = 1e-4
max_iter = 100000
best_acc_delay = 3000



arguments = docopt(__doc__)
print ("...Docopt... ")
print(arguments)
print ("............\n")
f_data_config = arguments['<f_data_config>']
f_model_config = arguments['<f_model_config>']

data_config = yaml.load(open(f_data_config, 'rb'))
model_config = yaml.load(open(f_model_config, 'rb'))
model_name = os.path.basename(f_model_config).split('.')[0]
data_name = os.path.basename(f_data_config).split('.')[0]
exp_name = '%s-X-%s-X-%s' % (model_name, data_name,arguments['<loss>'])
if arguments['--prefix']:
    exp_name = arguments['<p>']+':'+exp_name
fold_index = int(arguments['<fold_index>'])
testing = arguments['--test']
loss_class=  eval(arguments['<loss>'])()


# train(data_config, model_config, exp_name, fold_index,
#       init_lr, max_iter, best_acc_delay, eval(arguments['<loss>'])(),testing)

# def train(data_config, model_config, exp_name, fold_index, init_lr, max_iter, best_acc_delay, loss_class, testing=False):


# Initialize dataset/loader
dataset = BaseDataset(data_config, fold_index, load_raw=False)
extractor = eval(data_config['extractor_class'])(data_config)
if 'negative_fraction_hard' in data_config:
    nfh = data_config['negative_fraction_hard']
else:
    nfh = 0

cloader = Seq2SeqLoader(dataset, extractor, data_config[
    'batch_size'], fraction_positive=0.5,
    negative_fraction_hard=nfh, move_N_neg_to_val=1000)

with tf.variable_scope("main_model") as vs:
    net = eval(model_config['class_name'])(model_config['model_config'])
    net.build()
    main_model_variables = [v for v in tf.all_variables() if v.name.startswith(vs.name)]

BATCH_SIZE = model_config['model_config']['batch_size']
real_data =  tf.placeholder(tf.float32,
                    [model_config['model_config']['batch_size'],
                     model_config['model_config']['decoder_time_size'],
                     model_config['model_config']['dec_output_dim']])
fake_data = net.sample_trajectory()


with tf.variable_scope("disc_net") as vs:
    disc_net = eval(model_config['class_name'])(model_config['model_config'])
    disc_net_variables = [v for v in tf.all_variables() if v.name.startswith(vs.name)]

def _build_disc(disc_net, input_data):
    disc_out = disc_net.build(input_data)#[:, -1] TODO/TOTHINK: we could design output layer of disc differently
    return tf.reduce_mean(tf.reshape(disc_out, (model_config['model_config']['batch_size'], -1)), -1)

disc_real = _build_disc(disc_net, real_data)
disc_fake = _build_disc(disc_net, fake_data)

gen_params = main_model_variables
disc_params = disc_net_variables


if MODE == 'wgan':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(disc_cost, var_list=disc_params)

    clip_ops = []
    for var in disc_params:
        clip_bounds = [-.01, .01]
        clip_ops.append(
            tf.assign(
                var, 
                tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
            )
        )
    clip_disc_weights = tf.group(*clip_ops)
elif MODE == 'wgan-gp':
    # Standard WGAN loss
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    # Gradient penalty
    alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1], 
        minval=0.,
        maxval=1.
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)
    gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost += LAMBDA*gradient_penalty

    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)

elif MODE == 'dcgan':
    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.ones_like(disc_fake)))
    disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.zeros_like(disc_fake)))
    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real, labels=tf.ones_like(disc_real)))
    disc_cost /= 2.

    gen_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(gen_cost,var_list=lib.params_with_name('Generator'))
    disc_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(disc_cost,var_list=lib.params_with_name('Discriminator.'))





# [Supervised loss]
y_ = tf.placeholder(tf.float32,
                    [model_config['model_config']['batch_size'],
                     model_config['model_config']['decoder_time_size'],
                     model_config['model_config']['dec_input_dim']])
sup_loss = loss_class.build_tf_loss(net.output(), y_)


learning_rate = tf.placeholder(tf.float32, [])
global_step = tf.Variable(0)
sup_train_step = optimize_loss(sup_loss, global_step, learning_rate,
                           optimizer=lambda lr: tf.train.RMSPropOptimizer(lr),
                           clip_gradients=0.01, variables=main_model_variables)

### [Monitoring]
def _add_content_to_dic(dic, new_name):
    """
    name: [Variable, Placeholder, AssignOp, emtpy list]
    """
    tfv = tf.Variable(tf.constant(0.0), trainable=False)
    tfpl = tf.placeholder(tf.float32, shape=[], name=new_name)
    tfop = tf.assign(tfv, tfpl, name='update_%s'%new_name)
    dic[new_name] = [tfv,
                     tfpl,
                     tfop,
                     []]

val_monitor_dic = {}
_add_content_to_dic(val_monitor_dic, 'v_rloss')
_add_content_to_dic(val_monitor_dic, 'v_tloss')
train_monitor_dic = {}
_add_content_to_dic(train_monitor_dic, 't_loss')
for k, v in val_monitor_dic.items()+train_monitor_dic.items():
    tf.summary.scalar(k, v[0])
merged = tf.summary.merge_all()


session = tf.InteractiveSession()
sess = session
tf.global_variables_initializer().run()


### [Logging]
log_folder = os.path.join('./gan-logs', exp_name)
# remove existing log folder for the same model.
if os.path.exists(log_folder):
    import shutil
    shutil.rmtree(log_folder)
tf_writer = tf.summary.FileWriter(
    log_folder, session.graph)

saver = tf.train.Saver()
best_saver = tf.train.Saver()
###

def data_next(load_f, conti=True):
    loaded = load_f()
    if loaded is not None:
        dec_input, dec_output, enc_input,(history,pid)= loaded
    else:
        if conti:
            cloader.reset()
            loaded = cloader.next()
            dec_input, dec_output, enc_input,(history,pid)= loaded
        else: ## don't loop
            return None
    if pid is not None:
        start_frame = history[:,pid, -1]
    else:
        start_frame = history[:,:,-1].reshape(history.shape[0],-1)
    return dec_input, dec_output, enc_input,start_frame

def prepare_traj(traj, dec_output, start_frame, net):
    if net.output_format == 'location': ### scale 
        traj = wrapper_concatenated_last_dim(scale_last_dim, traj, upscale=True)
        gt_traj = wrapper_concatenated_last_dim(scale_last_dim, dec_output, upscale=True)
    elif net.output_format == 'velocity':
        gt_traj = experpolate_position(start_frame, dec_output)[:,1:]
    else:
        raise NotImplementedError
    return traj, gt_traj
    
# [Train]
best_val_teacher_forced_loss = np.inf
best_val_real_loss = np.inf
best_not_updated = 0
lrv = init_lr
tfs = model_config['model_config']['decoder_time_size']


for iter_ind in tqdm(range(max_iter)):
    ## done?
    best_not_updated += 1
    ## scheduling
    if iter_ind>0 and iter_ind % 5000 == 0:
        tfs -= 5
    if iter_ind ==10000000:## currently not using
        lrv *= .1
    

    ### [Update Generator/Predictor]
    ## ADV
    dec_input, dec_output, enc_input,start_frame= data_next(cloader.next)
    feed_dict = net.input(dec_input, 
                            teacher_forcing_stop=np.max([1, tfs]),
                            enc_input=enc_input
                            )
    net.aux_feed_dict({'start_frame':start_frame}, feed_dict)
    feed_dict = disc_net.input(None, 
                            teacher_forcing_stop=np.max([1, tfs]),
                            enc_input=enc_input, ret_dict=feed_dict
                            )
    if iter_ind > 0:
        _ = session.run(gen_train_op, feed_dict=feed_dict)
    ## SUP
    feed_dict[y_] = dec_output
    feed_dict[learning_rate] = lrv
    tl = sess.run(sup_train_step, feed_dict=feed_dict)
    train_monitor_dic['t_loss'][-1].append(tl)

    ## update Train Monitors
    # compute mean over valid batches
    for k, v in train_monitor_dic.items():
        v[-1] = np.mean(v[-1])
    # prepare feed_dict for tensorboard
    for k, v in train_monitor_dic.items():
        feed_dict[v[1]] = v[-1]
        v[-1] = []
    tmp = sess.run([v[2] for v in train_monitor_dic.values()] +[merged], feed_dict=feed_dict)
    tf_writer.add_summary(tmp[-1], iter_ind)  

    ### [Update Loss (i.e. discriminator)]
    if MODE == 'dcgan':
        disc_iters = 1
    else:
        disc_iters = CRITIC_ITERS
    for i in xrange(disc_iters):
        dec_input, dec_output, enc_input,start_frame= data_next(cloader.next)
        traj, gt_traj= prepare_traj(None, dec_output, start_frame, net)
        ### setup input for disc_net
        feed_dict = disc_net.input(dec_input, 
                            teacher_forcing_stop=np.max([1, tfs]),
                            enc_input=enc_input
                            )
        feed_dict[real_data] = gt_traj
        _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict=feed_dict)
        if MODE == 'wgan':
            _ = session.run(clip_disc_weights)


    # Validate
    if iter_ind % 10 == 0:
        ## loop throught valid examples
        while True:
            try:
                dec_input, dec_output, enc_input,start_frame= data_next(cloader.load_valid, conti=False)
            except TypeError: # None was returned
                break
            ## real-loss
            feed_dict = net.input(dec_input, 
                            teacher_forcing_stop=1,
                            enc_input=enc_input,
                            enc_keep_prob = 1.,
                            decoder_noise_level = 0.,
                            decoder_input_keep_prob = 1.
                            )
            feed_dict[y_] = dec_output
            val_loss = sess.run(sup_loss, feed_dict = feed_dict)
            val_monitor_dic['v_rloss'][-1].append(val_loss)
            
            ## traj-loss
            net.aux_feed_dict({'start_frame':start_frame}, feed_dict)
            traj = sess.run(net.sample_trajectory(), feed_dict = feed_dict)
            traj, gt_traj= prepare_traj(traj, dec_output, start_frame, net)
            l_traj = dist_trajectory(traj, gt_traj)
            val_monitor_dic['v_tloss'][-1].append(l_traj)
        
        # compute mean over valid batches
        for k, v in val_monitor_dic.items():
            v[-1] = np.mean(v[-1])
        # print to screen
        print_str = '[Iter: %g] '%(iter_ind)
        for k, v in val_monitor_dic.items():
            print_str += '| %s: %g'%(k, v[-1])
        print (print_str)
        # prepare feed_dict for tensorboard
        for k, v in val_monitor_dic.items():
            feed_dict[v[1]] = v[-1]
            v[-1] = []
        # update tensorboard
        tmp = sess.run([v[2] for v in val_monitor_dic.values()] +[merged], feed_dict=feed_dict)
        tf_writer.add_summary(tmp[-1], iter_ind)
        # Best model?
        # if val_real_loss < best_val_real_loss:
        #     best_not_updated = 0
        #     p = os.path.join("./gan-saves/", exp_name + '.ckpt.best')
        #     print ('Saving Best Model to: %s' % p)
        #     save_path = best_saver.save(sess, p)
        #     best_val_real_loss = val_real_loss
    if iter_ind % 2000 == 0:
        save_path = saver.save(sess, os.path.join(
            "./gan-saves/", exp_name + '%d.ckpt' % iter_ind))
    if best_not_updated == best_acc_delay:
        break
