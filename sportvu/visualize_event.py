"""visualize_event.py

Usage:
    visualize_event.py <fold_index> <f_data_config> <f_model_config> <f_anim_config>

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
from sportvu.data.loader import Seq2SeqLoader, BaseLoader
# concurrent
from resnet.utils.concurrent_batch_iter import ConcurrentBatchIterator
from tqdm import tqdm
from docopt import docopt
import yaml
import gc
from utils import truncated_mean, experpolate_position
from vis_utils import (make_sequence_prediction_image,make_sequence_sample_image,
                        repeat_first_dim_K_time, compute_euclidean_variance)
import cPickle as pkl
from sportvu.vis.Event import Event
from sportvu.anim.utils import find_position_idx_by_name
# import matplotlib.pylab as plt
# plt.ioff()
# fig = plt.figure()

arguments = docopt(__doc__)
print ("...Docopt... ")
print(arguments)
print ("............\n")
f_data_config = arguments['<f_data_config>']
f_model_config = arguments['<f_model_config>']
f_anim_config = arguments['<f_anim_config>']


data_config = yaml.load(open(f_data_config, 'rb'))
model_config = yaml.load(open(f_model_config, 'rb'))
anim_config = yaml.load(open(f_anim_config, 'rb'))
model_name = os.path.basename(f_model_config).split('.')[0]
data_name = os.path.basename(f_data_config).split('.')[0]
exp_name = '%s-X-%s' % (model_name, data_name)
fold_index = int(arguments['<fold_index>'])

data_config['batch_size'] = anim_config['n_samples']
# Initialize dataset/loader
dataset = BaseDataset(data_config, fold_index, load_raw=True)
extractor = eval(data_config['extractor_class'])(data_config)
if 'negative_fraction_hard' in data_config:
    nfh = data_config['negative_fraction_hard']
else:
    nfh = 0
baseloader = BaseLoader(dataset, extractor, data_config[
                        'n_negative_examples'], fraction_positive=0)
# loader = Seq2SeqLoader(dataset, extractor, data_config[
#     'batch_size'], fraction_positive=0.5,
#     negative_fraction_hard=nfh, move_N_neg_to_val=1000)

# cloader = loader

### get the targeted event 
event = Event(dataset.games[anim_config['gameid']]['events'][anim_config['event_id']],anim_config['gameid']) 
player_id = find_position_idx_by_name(anim_config['player_name'], event)
player_id = player_id + 1 ## ball
## prepare data for event/player
loaded = baseloader.load_event(game_id=anim_config['gameid'], event_id=anim_config['event_id'],
    every_K_frame=anim_config['every_K_frame'], player_id=player_id)
ret_val, ret_gameclocks, ret_frame_idx = loaded
dec_input, dec_output, enc_input, (meta)  = ret_val
history, pid = meta

gt_future_old = experpolate_position(history[:,pid,-1], dec_output)
# pkl.dump((ret_frame_idx, gt_future), open(anim_config['tmp_pkl'],'w'))
### adjust model spec
model_config['batch_size'] = anim_config['n_samples']
model_config['model_config']['batch_size'] = anim_config['n_samples']
model_config['model_config']['encoder_input_shape'][0] = anim_config['n_samples']
net = eval(model_config['class_name'])(model_config['model_config'])
net.build()

saver = tf.train.Saver()
sess = tf.InteractiveSession()
ckpt_path = os.path.join("./saves/", exp_name + '.ckpt.best')
saver.restore(sess, ckpt_path)


## teacher-forced loss
vs = []
imgs = []
pred_futures = []
for sample_ind in range(dec_input.shape[0]):
    feed_dict = net.input(repeat_first_dim_K_time(dec_input, K=data_config['batch_size'], ind=sample_ind), 
                    teacher_forcing_stop=None,
                    enc_input=repeat_first_dim_K_time(enc_input, K=data_config['batch_size'], ind=sample_ind),
                    enc_keep_prob = 1.,
                    decoder_noise_level = None,
                    decoder_input_keep_prob = None
                    )
    ### plot
    pred = sess.run(net.output(), feed_dict = feed_dict)
    m1_frame =repeat_first_dim_K_time( history[:,pid,-1], K=data_config['batch_size'], ind=sample_ind)
    gt_future = experpolate_position(m1_frame, repeat_first_dim_K_time( dec_output, K=data_config['batch_size'], ind=sample_ind))
    pred_future = experpolate_position(m1_frame, pred) # (B, T+1, 2)
    v = compute_euclidean_variance(pred_future)
    # imgs.append(make_sequence_sample_image(history[sample_ind][None], gt_future[0][None], pred_future, pid))
    vs.append(v)
    pred_futures.append(pred_future)
pkl.dump((ret_frame_idx, gt_future_old, np.array(pred_futures)), open(anim_config['tmp_pkl'],'wb'))