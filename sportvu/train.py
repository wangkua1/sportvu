"""train.py

Usage:
    train.py <fold_index> <f_data_config> <f_model_config>
    train.py --test <fold_index> <f_data_config> <f_model_config> <every_K_frame>
    train.py --test  --train <fold_index> <f_data_config> <f_model_config> <every_K_frame>

Arguments:
    <f_data_config>  example ''data/config/train_rev0.yaml''
    <f_model_config> example 'model/config/conv2d-3layers.yaml'

Example:
    python train.py 0 rev3_1-bmf-25x25.yaml conv2d-3layers-25x25.yaml --test --train 5
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
# if os.environ['HOME'] == '/u/wangkua1':  # jackson guppy
#     sys.path.append('/u/wangkua1/toolboxes/resnet')
# else:
#     sys.path.append('/ais/gobi4/slwang/sports/sportvu/resnet')
#     sys.path.append('/ais/gobi4/slwang/sports/sportvu')
from sportvu.model.convnet2d import ConvNet2d
from sportvu.model.convnet3d import ConvNet3d
# data
from sportvu.data.dataset import BaseDataset
from sportvu.data.extractor import BaseExtractor
from sportvu.data.loader import PreprocessedLoader, EventLoader, SequenceLoader, BaseLoader
# configuration
import config as CONFIG
# sanity
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
# concurrent
# from resnet.utils.concurrent_batch_iter import ConcurrentBatchIterator
from tqdm import tqdm
from docopt import docopt
import yaml
import gc
import cPickle as pkl


def train(data_config, model_config, exp_name, fold_index, init_lr, max_iter, best_acc_delay, testing=False):
    # Initialize dataset/loader
    dataset = BaseDataset(data_config, fold_index, load_raw=False)
    extractor = BaseExtractor(data_config)
    if 'negative_fraction_hard' in data_config:
        nfh = data_config['negative_fraction_hard']
        pfh = 1 - nfh
    else:
        nfh = 0
        pfh = 0.5
    if 'version' in data_config['extractor_config'] and data_config['extractor_config']['version'] >= 2:
        loader = SequenceLoader(
            dataset,
            extractor,
            data_config['batch_size'],
            fraction_positive=pfh,
            negative_fraction_hard=nfh
        )
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
    # sanity(val_x, exp_name)
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
    weights = tf.trainable_variables()
    l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights]) * 0.001
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=net.output()))
    loss = tf.reduce_mean(cross_entropy + l2_loss)

    # optimize
    # train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    global_step = tf.Variable(0)
    learning_rate = tf.placeholder(tf.float32, [])
    # learning_rate = tf.train.exponential_decay(init_lr, global_step, 10000, 0.96, staircase=True)
    # train_step = optimize_loss(cross_entropy, global_step, learning_rate,
    #             optimizer=lambda lr: tf.train.AdamOptimizer(lr))
    # train_step = optimize_loss(cross_entropy, global_step, learning_rate,
    #             optimizer=lambda lr: tf.train.MomentumOptimizer(lr, .9))
    # train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cross_entropy, global_step=global_step)
    train_step = optimize_loss(loss, global_step, learning_rate, optimizer=lambda lr: tf.train.AdamOptimizer(lr, .9))

    # reporting
    correct_prediction = tf.equal(tf.argmax(net.output(), 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    predictions = tf.argmax(net.output(), 1)
    true_labels = tf.argmax(y_, 1)
    confusion_matrix = tf.confusion_matrix(labels=true_labels, predictions=predictions, num_classes=2)

    tp = tf.count_nonzero(predictions * true_labels)
    tn = tf.count_nonzero((predictions - 1) * (true_labels - 1))
    fp = tf.count_nonzero(predictions * (true_labels - 1))
    fn = tf.count_nonzero((predictions - 1) * true_labels)

    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fmeasure = (2 * precision * recall) / (precision + recall)

    # testing
    if testing:

        every_K_frame = int(arguments['<every_K_frame>'])
        plot_folder = '%s/%s' % (CONFIG.plots.dir, exp_name)
        if not os.path.exists('%s/pkl'%(plot_folder)):
            os.makedirs('%s/pkl'%(plot_folder))

        saver = tf.train.Saver()
        sess = tf.InteractiveSession()
        ckpt_path = '%s/%s.ckpt.best' % (CONFIG.saves.dir,exp_name)
        saver.restore(sess, ckpt_path)

        feed_dict = net.input(val_x, 1, False)
        feed_dict[y_] = val_t
        ce, val_accuracy = sess.run([cross_entropy, accuracy], feed_dict=feed_dict)
        print ('Best Validation CE: %f, Acc: %f' % (ce, val_accuracy))

        # test section
        dataset = BaseDataset(f_data_config, int(arguments['<fold_index>']), load_raw=True)
        extractor = BaseExtractor(f_data_config)
        loader = BaseLoader(dataset, extractor, data_config['batch_size'], fraction_positive=0.5)
        cloader = loader

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

            # save the raw predictions
            probs = probs[:, 1]
            pkl.dump([gameclocks, probs, labels], open('%s/pkl/%s-%i.pkl'%(plot_folder,split, ind), 'w'))
            pkl.dump(meta, open('%s/pkl/%s-meta-%i.pkl'%(plot_folder,split, ind), 'w'))
            ind += 1

        sys.exit(0)

    # checkpoints
    if not os.path.exists(CONFIG.saves.dir):
        os.mkdir(CONFIG.saves.dir)
    # tensorboard
    if not os.path.exists(CONFIG.logs.dir):
        os.mkdir(CONFIG.logs.dir)

    tf.summary.scalar('cross_entropy', cross_entropy)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuray', accuracy)
    tf.summary.scalar('Accuracy', accuracy)
    tf.summary.scalar('Precision', precision)
    tf.summary.scalar('Recall', recall)
    tf.summary.scalar('F-Score', fmeasure)
    tf.summary.histogram('label_distribution', y_)
    tf.summary.histogram('logits', net.logits)

    if model_config['class_name'] == 'ConvNet2d':
        tf.summary.image('input', net.x, max_outputs=4)
    elif model_config['class_name'] == 'ConvNet3d':
        tf.summary.image('input', tf.reduce_sum(net.x, 1), max_outputs=4)
    else:
        raise Exception('input format not specified')

    merged = tf.summary.merge_all()
    log_folder = '%s/%s' % (CONFIG.logs.dir,exp_name)

    saver = tf.train.Saver()
    best_saver = tf.train.Saver()
    sess = tf.InteractiveSession()

    # remove existing log folder for the same model.
    if os.path.exists(log_folder):
        import shutil
        shutil.rmtree(log_folder)

    train_writer = tf.summary.FileWriter(os.path.join(log_folder, 'train'), sess.graph)
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
        if iter_ind % 3000 == 0 and iter_ind > 0:
            lrv *= .1
        feed_dict[learning_rate] = lrv
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict)
        train_writer.add_summary(summary, iter_ind)
        if iter_ind % 100 == 0:
            feed_dict = net.input(batch_xs, 1, False)
            feed_dict[y_] = batch_ys
            train_accuracy = accuracy.eval(feed_dict=feed_dict)
            train_ce = cross_entropy.eval(feed_dict=feed_dict)
            train_confusion_matrix = confusion_matrix.eval(feed_dict=feed_dict)

            # validate trained model
            feed_dict = net.input(val_x, 1, False)
            feed_dict[y_] = val_t
            summary, val_ce, val_accuracy = sess.run(
                [merged, cross_entropy, accuracy], feed_dict=feed_dict)
            val_writer.add_summary(summary, iter_ind)
            print("step %d, training accuracy %g, validation accuracy %g, train ce %g,  val ce %g" %
                  (iter_ind, train_accuracy, val_accuracy, train_ce, val_ce))
            # print('Train Confusion Matrix: \n %s' % (str(train_confusion_matrix)))

            if val_ce < best_val_ce:
                best_not_updated = 0
                p = '%s/%s.ckpt.best' % (CONFIG.saves.dir, exp_name)
                print ('Saving Best Model to: %s' % p)
                save_path = best_saver.save(sess, p)
                tf.train.export_meta_graph('%s.meta' % (p))
                best_val_ce = val_ce
        if iter_ind % 2000 == 0:
            save_path = saver.save(sess,'%s/%s-%d.ckpt'%(CONFIG.saves.dir,exp_name,iter_ind))
        if best_not_updated == best_acc_delay:
            break
    return best_val_ce


if __name__ == '__main__':
    arguments = docopt(__doc__)
    print ("...Docopt... ")
    print(arguments)
    print ("............\n")
    f_data_config = '%s/%s' % (CONFIG.data.config.dir,arguments['<f_data_config>'])
    f_model_config = '%s/%s' % (CONFIG.model.config.dir,arguments['<f_model_config>'])


    data_config = yaml.load(open(f_data_config, 'rb'))
    model_config = yaml.load(open(f_model_config, 'rb'))
    model_name = os.path.basename(f_model_config).split('.')[0]
    data_name = os.path.basename(f_data_config).split('.')[0]
    exp_name = '%s-X-%s' % (model_name, data_name)
    fold_index = int(arguments['<fold_index>'])
    init_lr = 1e-3
    max_iter = 30000
    best_acc_delay = 10000
    testing = arguments['--test']
    train(data_config, model_config, exp_name, fold_index, init_lr, max_iter, best_acc_delay, testing)
