from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# model
import tensorflow as tf

# data
from sportvu.data.dataset import BaseDataset
from sportvu.data.extractor import BaseExtractor
from sportvu.data.loader import BaseLoader
# concurrent
import sys
sys.path.append('/u/wangkua1/toolboxes/resnet')
from resnet.utils.concurrent_batch_iter import ConcurrentBatchIterator
from tqdm import tqdm


# Initialize dataset/loader
dataset = BaseDataset('data/config/train_rev0.yaml', fold_index=0)
extractor = BaseExtractor()
loader = BaseLoader(dataset, extractor, 32, fraction_positive=.5)
Q_size = 1000
N_thread = 32
cloader = ConcurrentBatchIterator(
    loader, max_queue_size=Q_size, num_threads=N_thread)

# model
x = tf.placeholder(tf.float32, [None, 50 * 100 * 3])
W = tf.Variable(tf.zeros([50 * 100 * 3, 2]))
b = tf.Variable(tf.zeros([2]))
y = tf.matmul(x, W) + b

y_ = tf.placeholder(tf.float32, [None, 2])

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# Train
for _ in tqdm(range(1000)):
    batch_xs, batch_ys = cloader.next()
    batch_xs = batch_xs.reshape(batch_xs.shape[0], -1)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: batch_xs,
                                        y_: batch_ys}))
