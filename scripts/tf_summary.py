
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
# model
import tensorflow as tf
import sys
import os
from tqdm import tqdm
from docopt import docopt
import yaml
import gc


x = tf.placeholder(tf.float32, [10,2])
y = tf.placeholder(tf.float32, [10,2])
euclid_loss = tf.reduce_sum(tf.pow(x-y, 2))
tf.summary.scalar('euclid_loss', euclid_loss)

merged = tf.summary.merge_all()
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
    
feed_dict = {x:np.random.rand(10,2),y:np.random.rand(10,2)}
summary, loss = sess.run([merged, euclid_loss], feed_dict=feed_dict)
