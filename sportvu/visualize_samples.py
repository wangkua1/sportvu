"""visualize_samples.py

Usage:
    visualize_samples.py <fold_index> <f_data_config> <f_model_config>

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
from vis_utils import make_sequence_prediction_image,make_sequence_sample_image
import cPickle as pkl
# import matplotlib.pylab as plt
# plt.ioff()
# fig = plt.figure()
def repeat_first_dim_K_time(x, ind=0, K=100):
    shape = [K] + [1 for i in range(len(x.shape)-1)]
    return np.tile(x[ind:ind+1], shape)

def compute_euclidean_variance(sequences):
    """
    (B, T+1, 2) -> 
    """
    mean = np.mean(sequences, axis=0)[None] #(1,T+1,2)
    return np.power(np.power(sequences - np.repeat(mean, sequences.shape[0], axis=0),2).sum(-1),.5).mean()

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
testing = False

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

saver = tf.train.Saver()
sess = tf.InteractiveSession()
ckpt_path = os.path.join("./saves/", exp_name + '.ckpt.best')
saver.restore(sess, ckpt_path)


loaded = cloader.load_valid()
if loaded is not None:
    dec_input, dec_output, enc_input, (meta)  = loaded

    history, pid = meta
# else: ## done
#     # print ('...')
#     break
## teacher-forced loss
vs = []
imgs = []
for sample_ind in range(data_config['batch_size']):
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
    imgs.append(make_sequence_sample_image(history[sample_ind][None], gt_future[sample_ind][None], pred_future, pid))
    vs.append(v)
pkl.dump(imgs, open('tmp.pkl','wb'))
# imgs = make_sequence_prediction_image(history, gt_future, pred_future, pid)
# pkl.dump((history, gt_future, pred_future, pid),
#           open(os.path.join("./logs/"+exp_name, 'iter-%g.pkl'%(iter_ind)),'wb'))

# for i in xrange(5):
#     plt.imshow(imgs[i])
#     plt.savefig(os.path.join("./saves/", exp_name +'iter-%g-%g.png'%(iter_ind,i)))


