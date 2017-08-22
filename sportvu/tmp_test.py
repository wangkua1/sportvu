"""tmp_test.py

Usage:
    tmp_test.py --test <fold_index> <f_data_config> <f_model_config> <n_samples>

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
if os.environ['HOME'] == '/u/wangkua1':  # jackson guppy
    sys.path.append('/u/wangkua1/toolboxes/resnet')
else:
    sys.path.append('/ais/gobi4/slwang/sports/sportvu/resnet')
    sys.path.append('/ais/gobi4/slwang/sports/sportvu')
from sportvu.model.seq2seq import Seq2Seq
from sportvu.model.encdec import EncDec
# data
from sportvu.data.dataset import BaseDataset
from sportvu.data.extractor import Seq2SeqExtractor, EncDecExtractor
from sportvu.data.loader import Seq2SeqLoader
# concurrent
from resnet.utils.concurrent_batch_iter import ConcurrentBatchIterator
from tqdm import tqdm
from docopt import docopt
import yaml
import gc
from utils import truncated_mean, experpolate_position
from vis_utils import make_sequence_prediction_image
import cPickle as pkl
from vis_utils import compute_euclidean_variance


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
exp_name = '%s-X-%s' % (model_name, data_name)
fold_index = int(arguments['<fold_index>'])
init_lr = 1e-4
max_iter = 100000
best_acc_delay = 3000
testing = arguments['--test']
# Initialize dataset/loader
dataset = BaseDataset(data_config, fold_index, load_raw=False)
extractor = eval(data_config['extractor_class'])(data_config)
if 'negative_fraction_hard' in data_config:
    nfh = data_config['negative_fraction_hard']
else:
    nfh = 0

loader = Seq2SeqLoader(dataset, extractor, data_config[
    'batch_size'], fraction_positive=0.5,
    negative_fraction_hard=nfh, move_N_neg_to_val=1000)
Q_size = 100
N_thread = 4
# cloader = ConcurrentBatchIterator(
#     loader, max_queue_size=Q_size, num_threads=N_thread)
cloader = loader

net = eval(model_config['class_name'])(model_config['model_config'])
net.build()

# build loss
y_ = tf.placeholder(tf.float32,
                    [model_config['model_config']['batch_size'],
                     model_config['model_config']['decoder_time_size'],
                     2])
learning_rate = tf.placeholder(tf.float32, [])
# euclid_loss = tf.reduce_mean(tf.pow(net.output() - y_, 2))
euclid_loss = tf.reduce_mean(
    tf.pow(tf.reduce_sum(tf.pow(net.output() - y_, 2), axis=-1), .5))
N_samples = int(arguments['<n_samples>'])
# testing
all_preds = []
if testing:
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    ckpt_path = os.path.join("./saves/", exp_name + '.ckpt.best')
    saver.restore(sess, ckpt_path)
    losses = []
    abs_losses = []
    variances = []
    while True:
        loaded = cloader.load_valid()
        if loaded is not None:
            dec_input, dec_output, enc_input, (meta) = loaded
            history, pid = meta
        else:  # done
            # print ('...')
            break
        gt_futures = []
        pred_futures = []
        for sample_ind in xrange(N_samples):
            # real-loss
            feed_dict = net.input(dec_input,
                                  teacher_forcing_stop=1,
                                  enc_input=enc_input,
                                  enc_keep_prob=1.,
                                  decoder_noise_level=None,
                                  decoder_input_keep_prob=None
                                  )
            feed_dict[y_] = dec_output
            val_loss = sess.run(euclid_loss, feed_dict=feed_dict)

            pred = sess.run(net.output(), feed_dict=feed_dict)
            gt_future = experpolate_position(history[:, pid, -1], dec_output)
            pred_future = experpolate_position(history[:, pid, -1], pred)
            gt_futures.append(gt_future)
            pred_futures.append(pred_future)
        pred_futures = np.array(pred_futures) #(S, B, T+1, 2)
        mean_pred = np.mean(pred_futures, axis=0)[None] #(1,B,T+1,2)
        var_pred = np.power(np.power(pred_futures - np.repeat(mean_pred, pred_futures.shape[0], axis=0),2).sum(-1),.5).mean(0).mean(-1) #(B,)
        abs_loss = np.power(np.power(mean_pred[0] - gt_future, 2).sum(-1), .5).mean(-1)
        losses.append(val_loss)
        abs_losses += list(abs_loss)
        variances += list(var_pred)
        all_preds.append(pred_futures)
    if N_samples == 1:
        weights = np.ones((len(variances)))
    else:
        weights = 1./np.array(variances)
        weights = weights / weights.sum() * len(weights)
    print ('Best Validation D(dx,dy): %f, D(x,y): %f, AD(x,y): %f' % (np.mean(losses), np.mean(abs_losses), np.mean(weights * np.array(abs_losses))))
    all_preds= np.concatenate(all_preds, axis=1) #(S, N, T+1, 2)
    pkl.dump((all_preds, np.array(abs_losses), np.array(weights * np.array(abs_losses))),
            open('./pkl/%s.pkl'%exp_name, 'wb'))
    # sys.exit(0)
####
# """
# notes for analyses across models
# 1. fix the validation set
# gt = (N, )
# history = 
# dec_input = (N, )
# """