

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
from vis_utils import (_pictorialize_single_sequence,make_1_channel_sampled_images,
                      make_3_channel_images)


f_data_config = 'data/config/rev3-ed-full-2d-2x.yaml'
data_config = yaml.load(open(f_data_config, 'rb'))
fold_index = 0
testing = True
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

# # testing
all_dec_inp = []
all_gt_out = []
all_hist = []
all_pid = []
while True:
    loaded = cloader.load_valid()
    if loaded is not None:
        dec_input, dec_output, enc_input, (meta) = loaded
        history, pid = meta
    else:  # done
        # print ('...')
        break
    all_dec_inp.append(dec_input)
    all_gt_out.append(dec_output)
    all_hist.append(history)
    all_pid.append(pid)
all_ps = []
all_absls =[]
all_adjls = []
# exp_names  = ['dec-128-X-rev3-ed-target-history-2x',
#               'ed-full-2d-128-X-rev3-ed-full-2d-2x',
#               ]
# exp_names = filter(lambda s:'-2x' in s, os.listdir('./pkl'))
exp_names = [
             'dec-128-X-rev3-ed-target-history-2x.pkl'
            , 'ed-target-history-128-X-rev3-ed-target-history-2x.pkl'
            , 'edd-target-history-dropout-128-X-rev3-ed-target-history-2x.pkl'
            , 'ed-full-2d-128-X-rev3-ed-full-2d-2x.pkl'
            , 'edd-full-2d-dropout-128-X-rev3-ed-full-2d-2x.pkl'
            ]
print (exp_names)
N = 1000
# S = 10
for exp_name in exp_names:
    (all_preds, abs_l, adj_l) = pkl.load(
            open('./pkl/%s'%exp_name, 'rb'))
    ## little hack
    assert (all_preds.shape[0] in [1,10]) ## hack
    if all_preds.shape[0] == 1:
        all_preds = all_preds.repeat(10,0)
    all_ps.append(all_preds[:,:N])
    all_absls.append(abs_l[:N])
    all_adjls.append(adj_l[:N])
all_hist = np.concatenate(all_hist,0)[:N]
all_dec_inp = all_hist[:,1]
all_dec_out = np.concatenate(all_gt_out,0)[:N]
all_dec_out = experpolate_position(all_hist[:,1,-1], all_dec_out)
all_preds_old = np.array(all_ps) #(M,S,N,T,2)
import matplotlib.pylab as plt
plt.ioff()


def plot(all_hist, all_dec_inp, all_dec_out, all_preds, all_inds, dir_name='random'):
    batch_size = 5
    N_batches = len(all_inds)//batch_size
    for batch_ind in xrange(N_batches):
        curr_inds = np.arange(batch_ind*batch_size,(batch_ind+1) * batch_size)
        ### make all the images
        # history images
        hist_imgs = make_3_channel_images(all_hist[all_inds[curr_inds]])
        ## for prediction images,
        ## make 3 channels separately
        # decoder-inputs
        dec_inp_channels = _pictorialize_single_sequence(all_dec_inp[all_inds[curr_inds]])
        # decoder-outputs
        dec_out_channels = _pictorialize_single_sequence(all_dec_out[all_inds[curr_inds]])
        # decoder-predictions
        all_preds = np.transpose(all_preds_old[:,:,all_inds[curr_inds]], (0,2,1,3,4)) # (M,N,S,T,2)
        old_shape = list(all_preds.shape)
        all_preds = np.reshape(all_preds, [-1] + old_shape[-3:])
        pred_channels  = make_1_channel_sampled_images(all_preds) #(M*N, X,Y)
        pred_channels = np.reshape(pred_channels, old_shape[:2]+list(pred_channels.shape)[-2:])#(M,N,X,Y)

        for img_ind in tqdm(xrange(batch_size)):
            imgs = [hist_imgs[img_ind]]
            for model_ind in xrange(pred_channels.shape[0]):
                pred_img = np.zeros(list(pred_channels.shape)[-2:]+[3])
                pred_img[...,0] = dec_inp_channels[img_ind] *.8
                pred_img[...,1] = dec_out_channels[img_ind] *.8
                pred_img[...,2] = np.power(pred_channels[model_ind, img_ind],1)
                imgs.append(pred_img)
            final_img = np.concatenate(imgs, 1)
            plt.imshow(final_img)
            if not os.path.exists('imgs/%s'%dir_name):
                os.makedirs('imgs/%s'%dir_name)
            plt.savefig('imgs/%s/%g.png'%(dir_name, batch_size*batch_ind+img_ind))

plot(all_hist,all_dec_inp, all_dec_out, all_preds, np.arange(0,20))
####some analysis
top_K = 20
all_adjls = np.array(all_adjls)
tmp_inds = np.argsort(all_adjls.mean(0))
## all does well
plot(all_hist, all_dec_inp, all_dec_out, all_preds, tmp_inds[:top_K], 'all_good')
## all does poorly
plot(all_hist, all_dec_inp, all_dec_out, all_preds, tmp_inds[-top_K:], 'all_bad')
## for each one
for exp_ind, exp_name in enumerate(exp_names):
    exp_name = exp_name.split('.')[0]
    all_adjls = np.array(all_adjls)
    cur_exp_adjls = all_adjls[exp_ind]
    other_exp_adjls = np.concatenate([all_adjls[:exp_ind], all_adjls[exp_ind+1:]], 0).mean(0)
    
    tmp_inds = np.argsort(cur_exp_adjls - other_exp_adjls)
    ## this does well
    plot(all_hist, all_dec_inp, all_dec_out, all_preds, tmp_inds[:top_K], '%s_good'%(exp_name))
    ## this does poorly
    plot(all_hist, all_dec_inp, all_dec_out, all_preds, tmp_inds[-top_K:], '%s_bad'%(exp_name))


# vs = []
# imgs = []
# pred_futures = []
# for sample_ind in range(dec_input.shape[0]):
#     feed_dict = net.input(repeat_first_dim_K_time(dec_input, K=data_config['batch_size'], ind=sample_ind), 
#                     teacher_forcing_stop=None,
#                     enc_input=repeat_first_dim_K_time(enc_input, K=data_config['batch_size'], ind=sample_ind),
#                     enc_keep_prob = 1.,
#                     decoder_noise_level = None,
#                     decoder_input_keep_prob = None
#                     )
#     ### plot
#     pred = sess.run(net.output(), feed_dict = feed_dict)
#     m1_frame =repeat_first_dim_K_time( history[:,pid,-1], K=data_config['batch_size'], ind=sample_ind)
#     gt_future = experpolate_position(m1_frame, repeat_first_dim_K_time( dec_output, K=data_config['batch_size'], ind=sample_ind))
#     pred_future = experpolate_position(m1_frame, pred) # (B, T+1, 2)
#     v = compute_euclidean_variance(pred_future)