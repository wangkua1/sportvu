
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
# data
from sportvu.data.dataset import BaseDataset
from sportvu.data.extractor import Seq2SeqExtractor
from sportvu.data.loader import Seq2SeqLoader
# concurrent
from resnet.utils.concurrent_batch_iter import ConcurrentBatchIterator
from tqdm import tqdm
from docopt import docopt
import yaml
import gc

f_data_config = '../sportvu/data/config/rev3-s2s.yaml'
data_config = yaml.load(open(f_data_config, 'rb'))

# Initialize dataset/loader
dataset = BaseDataset(data_config, 0, load_raw=True)
extractor = Seq2SeqExtractor(data_config)
if 'negative_fraction_hard' in data_config:
    nfh = data_config['negative_fraction_hard']
else:
    nfh = 0

loader = Seq2SeqLoader(dataset, extractor, data_config[
    'batch_size'], fraction_positive=0.5,
    negative_fraction_hard=nfh)

def print_mean_error(loader):
    l = []
    for i in tqdm(xrange(100)):
        loaded = loader.next()
        if loaded is not None:
            enc_input, dec_input, dec_target_sequence, dec_output = loaded
        else:
            loader.reset()
            continue
        l.append(np.power(dec_output, 2).mean())
    print( np.mean(l) )
       