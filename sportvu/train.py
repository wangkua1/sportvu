"""train.py

Usage:
    train.py <fold_index> <f_data_config> <f_model_config> 
    train.py --test <fold_index> <f_data_config> <f_model_config>

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
from sportvu.model.convnet2d import ConvNet2d
from sportvu.model.convnet3d import ConvNet3d
# data
from sportvu.data.dataset import BaseDataset
from sportvu.data.extractor import BaseExtractor
from sportvu.data.loader import PreprocessedLoader, EventLoader, SequenceLoader
# concurrent
from resnet.utils.concurrent_batch_iter import ConcurrentBatchIterator
from tqdm import tqdm
from docopt import docopt
import yaml
import gc



def train(data_config, model_config, exp_name, fold_index, init_lr, max_iter, best_acc_delay, testing=False):
    # Initialize dataset/loader
    dataset = BaseDataset(data_config, fold_index, load_raw=True)
    extractor = BaseExtractor(data_config)
    if 'negative_fraction_hard' in data_config:
        nfh = data_config['negative_fraction_hard']
    else:
        nfh = 0
    if ('version' in data_config['extractor_config']
            and data_config['extractor_config']['version'] >= 2):
        loader = SequenceLoader(dataset, extractor, data_config[
                                'batch_size'], fraction_positive=0.5,
                                negative_fraction_hard=nfh)
    elif 'no_extract' in data_config and data_config['no_extract']:
        loader = EventLoader(dataset, extractor, None, fraction_positive=0.5)
    else:
        loader = PreprocessedLoader(
            dataset, extractor, None, fraction_positive=0.5)
    Q_size = 100
    N_thread = 4
    # cloader = ConcurrentBatchIterator(
    #     loader, max_queue_size=Q_size, num_threads=N_thread)
    cloader = loader


    val_x, val_t = loader.load_valid()
    if model_config['class_name'] == 'ConvNet2d':
        val_x = np.rollaxis(val_x, 1, 4)
    elif model_config['class_name'] == 'ConvNet3d':
        val_x = np.rollaxis(val_x, 1, 5)
    else:
        raise Exception('input format not specified')

    net = eval(model_config['class_name'])(model_config['model_config'])
    net.build()

    # build loss
    y_ = tf.placeholder(tf.float32, [None, 2])
    learning_rate = tf.placeholder(tf.float32, [])
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=net.output()))
    # train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    global_step = tf.Variable(0)
    # train_step = optimize_loss(cross_entropy, global_step, learning_rate, 
    #             optimizer=lambda lr: tf.train.AdamOptimizer(lr))
    train_step = optimize_loss(cross_entropy, global_step, learning_rate, 
                optimizer=lambda lr: tf.train.MomentumOptimizer(lr, .9))
    correct_prediction = tf.equal(tf.argmax(net.output(), 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    # testing
    if testing:
        saver = tf.train.Saver()
        sess = tf.InteractiveSession()
        ckpt_path = os.path.join("./saves/", exp_name + '.ckpt.best')
        saver.restore(sess, ckpt_path)

        feed_dict = net.input(val_x, 1, False)
        feed_dict[y_] = val_t
        ce, val_accuracy = sess.run([cross_entropy, accuracy], feed_dict=feed_dict)
        print ('Best Validation CE: %f, Acc: %f' % (ce, val_accuracy))
        sys.exit(0)

    # checkpoints
    if not os.path.exists('./saves'):
        os.mkdir('./saves')
    # tensorboard
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    tf.summary.scalar('cross_entropy', cross_entropy)
    tf.summary.scalar('accuray', accuracy)
    if model_config['class_name'] == 'ConvNet2d':
        tf.summary.image('input', net.x, max_outputs=4)
    elif model_config['class_name'] == 'ConvNet3d':
        tf.summary.image('input', tf.reduce_sum(net.x, 1), max_outputs=4)
    else:
        raise Exception('input format not specified')
    tf.summary.histogram('label_distribution', y_)
    tf.summary.histogram('logits', net.logits)

    merged = tf.summary.merge_all()
    log_folder = os.path.join('./logs', exp_name)

    saver = tf.train.Saver()
    sess = tf.InteractiveSession()

    # remove existing log folder for the same model.
    if os.path.exists(log_folder):
        import shutil
        shutil.rmtree(log_folder)

    train_writer = tf.summary.FileWriter(
        os.path.join(log_folder, 'train'), sess.graph)
    val_writer = tf.summary.FileWriter(os.path.join(log_folder, 'val'), sess.graph)
    tf.global_variables_initializer().run()
    # Train
    # best_val_acc = 0
    best_val_ce = np.inf
    best_not_updated = 0
    lrv = init_lr
    for iter_ind in tqdm(range(max_iter)):
        best_not_updated += 1
        # if iter_ind ==0:
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
        feed_dict = net.input(batch_xs, None, True)
        feed_dict[y_] = batch_ys
        if iter_ind ==max_iter//2:
            lrv *= .1
        feed_dict[learning_rate] = lrv
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict)
        train_writer.add_summary(summary, iter_ind)
        if iter_ind % 100 == 0:
            feed_dict = net.input(batch_xs, 1, False)
            feed_dict[y_] = batch_ys
            train_accuracy = accuracy.eval(feed_dict=feed_dict)
            # validate trained model
            feed_dict = net.input(val_x, 1, False)
            feed_dict[y_] = val_t
            summary, val_ce, val_accuracy = sess.run(
                [merged, cross_entropy, accuracy], feed_dict=feed_dict)
            val_writer.add_summary(summary, iter_ind)
            print("step %d, training accuracy %g, validation accuracy %g, val ce %g" %
                  (iter_ind, train_accuracy, val_accuracy, val_ce))
            if val_ce < best_val_ce:
                best_not_updated = 0
                p = os.path.join("./saves/", exp_name + '.ckpt.best')
                print ('Saving Best Model to: %s' % p)
                save_path = saver.save(sess, p)
                best_val_ce = val_ce
        if iter_ind % 2000 == 0:
            save_path = saver.save(sess, os.path.join(
                "./saves/", exp_name + '%d.ckpt' % iter_ind))
        if best_not_updated == best_acc_delay:
            break
    return best_val_ce


if __name__ == '__main__':
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
    init_lr = 1e-3
    max_iter = 10000
    best_acc_delay = 3000
    testing = arguments['--test']
    train(data_config, model_config, exp_name, fold_index, init_lr, max_iter, best_acc_delay, testing)