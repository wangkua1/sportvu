"""test.py

Usage:
    test.py <fold_index> <f_data_config> <f_model_config> <every_K_frame> --train
    test.py <fold_index> <f_data_config> <f_model_config> <every_K_frame>

Arguments:
    <f_data_config>  example 'train_rev0.yaml'
    <f_model_config> example 'conv2d-3layers.yaml'

Example:
    python test.py 0 rev3_1-bmf-25x25.yaml conv2d-3layers-25x25.yaml 5
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
# configuration
import config as CONFIG
# model
import tensorflow as tf
import sys
import os
if os.environ['HOME'] == '/u/wangkua1':  # jackson guppy
    sys.path.append('/u/wangkua1/toolboxes/resnet')
else:
    sys.path.append('/ais/gobi4/slwang/sports/sportvu/resnet')
    sys.path.append('/ais/gobi4/slwang/sports/sportvu')
from sportvu.model.convnet2d import ConvNet2d
from sportvu.model.convnet3d import ConvNet3d
# data
from sportvu.data.dataset import BaseDataset
from sportvu.data.extractor import BaseExtractor
from sportvu.data.loader import BaseLoader
# concurrent
# from resnet.utils.concurrent_batch_iter import ConcurrentBatchIterator
from tqdm import tqdm
from docopt import docopt
import yaml
import gc
import matplotlib.pylab as plt
import cPickle as pkl
arguments = docopt(__doc__)
print ("...Docopt... ")
print(arguments)
print ("............\n")

f_data_config = '%s/%s' % (CONFIG.data.config.dir,arguments['<f_data_config>'])
f_model_config = '%s/%s' % (CONFIG.model.config.dir,arguments['<f_model_config>'])
# pre_trained = arguments['<pre_trained>']
data_config = yaml.load(open(f_data_config, 'rb'))
model_config = yaml.load(open(f_model_config, 'rb'))
model_name = os.path.basename(f_model_config).split('.')[0]
data_name = os.path.basename(f_data_config).split('.')[0]
exp_name = '%s-X-%s' % (model_name, data_name)
# Initialize dataset/loader
dataset = BaseDataset(f_data_config, int(
    arguments['<fold_index>']), load_raw=True)
extractor = BaseExtractor(f_data_config)
loader = BaseLoader(dataset, extractor, data_config[
                    'batch_size'], fraction_positive=0.5)
Q_size = 100
N_thread = 4
# cloader = ConcurrentBatchIterator(
#     loader, max_queue_size=Q_size, num_threads=N_thread)
cloader = loader

net = eval(model_config['class_name'])(model_config['model_config'])
net.build()

# build loss
y_ = tf.placeholder(tf.float32, [None, 2])
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=net.output()))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(net.output(), 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# plot_folder = os.path.join('./plots', exp_name)
plot_folder = '%s/%s' % (CONFIG.plots.dir, exp_name)
if not os.path.exists('%s/pkl'%(plot_folder)):
    os.makedirs('%s/pkl'%(plot_folder))

ckpt_path = '%s/%s.ckpt.best' % (CONFIG.saves.dir,exp_name)
meta_path = ckpt_path + '.meta'
saver = tf.train.import_meta_graph(meta_path)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# ckpt_path = os.path.join("./saves/", exp_name + '.ckpt.best')
# ckpt_path = os.path.join("./saves/", exp_name + '10000.ckpt')
saver.restore(sess, ckpt_path)

every_K_frame = int(arguments['<every_K_frame>'])
# plt.figure()
ind = 0
while True:
    print (ind)
    if arguments['--train']:
        split = 'train'
    else:
        split = 'val'
    loaded = cloader.load_split_event(split, True, every_K_frame)
    if loaded is not None:
        if loaded == 0:
            ind+=1
            continue
        batch_xs, labels, gameclocks, meta = loaded
    else:
        print ('Bye')
        sys.exit(0)
    if model_config['class_name'] == 'ConvNet2d':
        batch_xs = np.rollaxis(batch_xs, 1, 4)
    elif model_config['class_name'] == 'ConvNet3d':
        batch_xs = np.rollaxis(batch_xs, 1, 5)
    else:
        raise Exception('input format not specified')
    feed_dict = net.input(batch_xs, None, True)
    probs = sess.run(tf.nn.softmax(net.output()), feed_dict=feed_dict)

    # plt.plot(gameclocks, probs[:, 1], '-')
    # plt.plot(np.array(labels), np.ones((len(labels))), '.')

    # plt.savefig(os.path.join(plot_folder, '%i.png' % ind))
    # plt.clf()
    # save the raw predictions
    pkl.dump([gameclocks, probs[:, 1], labels], open('%s/pkl/%s-%i.pkl'%(plot_folder,split, ind), 'w'))
    if arguments['--train']:
        pkl.dump(meta, open('%s/pkl/%s-meta-%i.pkl'%(plot_folder,split, ind), 'w'))
    ind += 1
