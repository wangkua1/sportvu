"""train.py

Usage:
    train.py <fold_index> <f_data_config> <f_model_config> 

Arguments:
    <f_data_config>  example ''data/config/train_rev0.yaml''
    <f_model_config> example 'model/config/conv2d-3layers.yaml'

Example:
    python train.py 0 data/config/train_rev0.yaml model/config/conv2d-3layers.yaml
    python train.py 0 data/config/train_rev0_vid.yaml model/config/conv3d-1.yaml
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
# model
import tensorflow as tf
from sportvu.model.convnet2d import ConvNet2d
from sportvu.model.convnet3d import ConvNet3d
# data
from sportvu.data.dataset import BaseDataset
from sportvu.data.extractor import BaseExtractor
from sportvu.data.loader import PreprocessedLoader
# concurrent
import sys
sys.path.append('/u/wangkua1/toolboxes/resnet')
from resnet.utils.concurrent_batch_iter import ConcurrentBatchIterator
from tqdm import tqdm
from docopt import docopt
import yaml

arguments = docopt(__doc__)
print ("...Docopt... ")
print(arguments)
print ("............\n")

f_data_config = arguments['<f_data_config>']
f_model_config = arguments['<f_model_config>']
model_config = yaml.load(open(f_model_config, 'rb'))
# Initialize dataset/loader
dataset = BaseDataset(f_data_config, int(arguments['<fold_index>']), load_raw=False)
extractor = BaseExtractor(f_data_config)
loader = PreprocessedLoader(dataset, extractor, None, fraction_positive=0.5)
Q_size = 100
N_thread = 32
val_x, val_t = loader.load_valid()
cloader = ConcurrentBatchIterator(
    loader, max_queue_size=Q_size, num_threads=N_thread)


net = eval(model_config['class_name'])(model_config['model_config'])
net.build()

# build loss
y_ = tf.placeholder(tf.float32, [None, 2])
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=net.output()))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(net.output(), 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# Train
for iter_ind in tqdm(range(20000)):
    loaded = cloader.next()
    if loaded is not None:
        batch_xs, batch_ys = loaded
    else:
        cloader.reset()
        continue
    if model_config['class_name'] == 'ConvNet2d':
        batch_xs = np.rollaxis(batch_xs, 1, 4)
    elif model_config['class_name'] == 'ConvNet3d':
        batch_xs = np.rollaxis(batch_xs, 1, 5)
    else:
        raise Exception('input format not specified')
    feed_dict = net.input(batch_xs)
    feed_dict[y_] = batch_ys
    train_step.run(feed_dict=feed_dict)
    if iter_ind % 100 == 0:
        feed_dict = net.input(batch_xs, 1)
        feed_dict[y_] = batch_ys    
        train_accuracy = accuracy.eval(feed_dict=feed_dict)
        print("step %d, training accuracy %g" % (iter_ind, train_accuracy))

        # validate trained model
        if model_config['class_name'] == 'ConvNet2d':
            val_x = np.rollaxis(val_x, 1, 4)
        elif model_config['class_name'] == 'ConvNet3d':
            val_x = np.rollaxis(val_x, 1, 5)
        else:
            raise Exception('input format not specified')
        feed_dict = net.input(val_x, 1)
        feed_dict[y_] = val_t    
        print(sess.run(accuracy, feed_dict=feed_dict))
        # print(sess.run(accuracy, feed_dict={x: val_x.reshape(val_x.shape[0], -1),
        #                                     y_: val_t}))
