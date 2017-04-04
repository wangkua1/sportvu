from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
# model
import tensorflow as tf

# data
from sportvu.data.dataset import BaseDataset
from sportvu.data.extractor import BaseExtractor
from sportvu.data.loader import PreprocessedLoader
# concurrent
import sys
sys.path.append('/u/wangkua1/toolboxes/resnet')
from resnet.utils.concurrent_batch_iter import ConcurrentBatchIterator
from tqdm import tqdm

f_data_config = 'data/config/train_rev0.yaml'
# Initialize dataset/loader
dataset = BaseDataset(f_data_config, 0, load_raw=False)
extractor = BaseExtractor(f_data_config)
loader = PreprocessedLoader(dataset, extractor, None, fraction_positive=0.5)
Q_size = 1000
N_thread = 32
val_x, val_t = loader.load_valid()
# cloader = ConcurrentBatchIterator(
#     loader, max_queue_size=Q_size, num_threads=N_thread)


# model
x = tf.placeholder(tf.float32, [None, 100, 50, 3])
keep_prob = tf.placeholder(tf.float32)

#convnet
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
## init weights/bias
W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
## build model
x_image = x
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
h_pool1_drop = tf.nn.dropout(h_pool1, keep_prob)
h_conv2 = tf.nn.relu(conv2d(h_pool1_drop, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
h_pool2_drop = tf.nn.dropout(h_pool2, keep_prob)

"""
shape
"""
SHAPE_convlast= 25*13*64
W_fc1 = weight_variable([SHAPE_convlast, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2_drop, [-1, SHAPE_convlast])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])


y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# y = tf.matmul(x, W) + b

y_ = tf.placeholder(tf.float32, [None, 2])

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# Train
for iter_ind in tqdm(range(20000)):
    loaded = loader.next()
    if loaded != None:
        batch_xs, batch_ys = loaded
    else: 
        loader.reset()
        continue
    # batch_xs = batch_xs.reshape(batch_xs.shape[0], -1)
    batch_xs = np.rollaxis(batch_xs, 1, 4)
    train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
    if iter_ind%100==0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch_xs, y_: batch_ys, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(iter_ind, train_accuracy))
          

        # validate trained model
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy, feed_dict={x: np.rollaxis(val_x,1,4),
                                            y_: val_t,
                                            keep_prob: 1.0}))
        # print(sess.run(accuracy, feed_dict={x: val_x.reshape(val_x.shape[0], -1),
        #                                     y_: val_t}))
