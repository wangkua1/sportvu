"""make_raw_from_untrained.py

Usage:
    make_raw_from_untrained.py <fold_index> <f_data_config> <f_model_config> <f_detect_config> <every_K_frame>

Arguments:
    <f_data_config> from contains config of game_ids that have not been trained example: rev3_1-bmf-25x25.yaml
    <f_model_config> example: conv2d-3layers-25x25.yaml
    <f_detect_config> example: nms1.yaml
    <game_id> game that was not in trained dataset, example: 0021500196
    --list all game that are not in trained dataset, example: 0021500188

Example:
     python make_raw_from_untrained.py 0 rev3_1-bmf-25x25.yaml conv2d-3layers-25x25.yaml nms1.yaml 5
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
optimize_loss = tf.contrib.layers.optimize_loss
import numpy as np
import os
import sys
from tqdm import tqdm
from docopt import docopt
import yaml
import gc
import cPickle as pkl
import matplotlib.pyplot as plt

from sportvu.model.convnet2d import ConvNet2d
from sportvu.model.convnet3d import ConvNet3d
from sportvu.data import constant
from sportvu.detect.nms import NMS
from sportvu.data.dataset import BaseDataset
from sportvu.data.extractor import BaseExtractor
from sportvu.data.loader import PreprocessedLoader, EventLoader, SequenceLoader, BaseLoader
import config as CONFIG
game_dir = constant.game_dir
pnr_dir = os.path.join(game_dir, 'pnr-annotations')

def make_raw(data_config, model_config, exp_name, fold_index, every_K_frame, plot_folder):
    """
    Load trained model and create probabilities from raw data that is seperated
    from training and testing data.

    Parameters
    ----------
    data_config: dict
        config that contains games trained on
    model_config: dict
        config
    exp_name: str
        config
    fold_index: int
        config
    every_K_frame: int
        splitting events
    plot_folder: str
        location of pkl files


    Returns
    -------
    probs: ndarray
        probabilities
    meta: array
        information about probabilities
    """

    dataset = BaseDataset(data_config, fold_index, load_raw=False)
    extractor = BaseExtractor(data_config)
    if 'negative_fraction_hard' in data_config:
        nfh = data_config['negative_fraction_hard']
    else:
        nfh = 0
    loader = SequenceLoader(
                                dataset,
                                extractor,
                                data_config['batch_size'],
                                fraction_positive=0.5,
                                negative_fraction_hard=nfh
                            )
    cloader = loader

    val_x, val_t = loader.load_valid()
    if model_config['class_name'] == 'ConvNet2d':
        val_x = np.rollaxis(val_x, 1, 4)
    elif model_config['class_name'] == 'ConvNet3d':
        val_x = np.rollaxis(val_x, 1, 5)

    net = eval(model_config['class_name'])(model_config['model_config'])
    net.build()

    # build loss
    y_ = tf.placeholder(tf.float32, [None, 2])
    learning_rate = tf.placeholder(tf.float32, [])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=net.output()))
    global_step = tf.Variable(0)
    train_step = optimize_loss(cross_entropy, global_step, learning_rate,
                optimizer=lambda lr: tf.train.MomentumOptimizer(lr, .9))
    correct_prediction = tf.equal(tf.argmax(net.output(), 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    ckpt_path = '%s/%s.ckpt.best' % (CONFIG.saves.dir,exp_name)
    saver.restore(sess, ckpt_path)

    # sanity from training
    feed_dict = net.input(val_x, 1, False)
    feed_dict[y_] = val_t
    ce, val_accuracy = sess.run([cross_entropy, accuracy], feed_dict=feed_dict)
    print ('Best Validation CE: %f, Acc: %f' % (ce, val_accuracy))

    # test section, change to games that have not been trained on
    data_config['data_config']['game_ids'] = data_config['data_config']['game_ids_no_train']
    dataset = BaseDataset(data_config, int(arguments['<fold_index>']), load_raw=True, no_anno=True)
    extractor = BaseExtractor(data_config)
    loader = BaseLoader(dataset, extractor, data_config['batch_size'], fraction_positive=0.5)

    ind = 0
    for game_id in dataset.game_ids:
        game_events = dataset.games[game_id]['events']
        for event_id, event in enumerate(game_events):
            event = event['playbyplay']
            if event.empty:
                continue

            print('Game: %s, Event: %i' % (game_id, event_id))
            loaded = loader.load_event(game_id=game_id, event_id=event_id, every_K_frame=every_K_frame)
            if loaded is not None:
                if loaded == 0:
                    ind+=1
                    continue
                batch_xs, gameclocks, labels = loaded
                meta = [event_id, game_id]

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
            pkl.dump([gameclocks, probs, labels], open('%s/pkl/raw-%i.pkl'%(plot_folder, ind), 'w'))
            pkl.dump(meta, open('%s/pkl/raw-meta-%i.pkl'%(plot_folder, ind), 'w'))
            ind += 1

def label_in_cand(cand, labels):
    for l in labels:
        if l > cand[1] and l < cand[0]:
            return True
    return False

def detect_from_prob(data_config, model_config, detect_config, exp_name, fold_index, plot_folder):
    """
    From probability maps, use detection to identify pnr instances from
    unseen data.

    Parameters
    ----------
    data_config: dict
        config
    model_config: dict
        config
    detect_config: dict
        config
    exp_name: str
        config
    fold_index: int
        config
    plot_folder: str
        location of pkl files

    Returns
    -------
    """
    dataset = BaseDataset(data_config, int(arguments['<fold_index>']), load_raw=True, no_anno=True)
    detector = eval(detect_config['class'])(detect_config)
    all_pred_f = filter(lambda s:'raw-' in s and 'raw-meta' not in s,os.listdir('%s/pkl'%(plot_folder)))

    annotations = []
    for _, f in tqdm(enumerate(all_pred_f)):
        ind = int(f.split('.')[0].split('-')[1])
        gameclocks, pnr_probs, labels = pkl.load(open('%s/pkl/raw-%i.pkl'%(plot_folder,ind), 'rb'))
        meta = pkl.load(open('%s/pkl/raw-meta-%i.pkl' %(plot_folder, ind), 'rb'))
        cands, mp, frame_indices = detector.detect(pnr_probs, gameclocks, True)

        plt.plot(gameclocks, pnr_probs, '-')
        if mp is not None:
            plt.plot(gameclocks, mp, '-')
        # plt.plot(np.array(labels), np.ones((len(labels))), '.')
        for ind, cand in enumerate(cands):
            cand_x = np.arange(cand[1], cand[0], .1)
            plt.plot(cand_x, np.ones((len(cand_x))) * .95, '-' )
            ## if FP, record annotations
            if not label_in_cand(cand, labels):
                anno = {'gameid':meta[1], 'gameclock':gameclocks[frame_indices[ind]],
                        'eid':meta[0], 'quarter':dataset.games[meta[1]]['events'][meta[0]]['quarter']}
                annotations.append(anno)

        plt.ylim([0,1])
        plt.title('Game: %s, Event: %i'%(meta[1], meta[0]))
        plt.savefig('%s/%s-raw-%i.png' %(plot_folder,detect_config['class'], ind))
        plt.clf()

    pkl.dump(annotations, open('%s/gt/from-raw-examples.pkl'%(pnr_dir), 'wb'))


if __name__ == '__main__':
    arguments = docopt(__doc__)
    print ("...Docopt... ")
    print(arguments)
    print ("............\n")

    f_data_config = '%s/%s' % (CONFIG.data.config.dir,arguments['<f_data_config>'])
    f_model_config = '%s/%s' % (CONFIG.model.config.dir,arguments['<f_model_config>'])
    f_detect_config = '%s/%s' % (CONFIG.detect.config.dir,arguments['<f_detect_config>'])

    data_config = yaml.load(open(f_data_config, 'rb'))
    model_config = yaml.load(open(f_model_config, 'rb'))
    detect_config = yaml.load(open(f_detect_config, 'rb'))

    model_name = os.path.basename(f_model_config).split('.')[0]
    data_name = os.path.basename(f_data_config).split('.')[0]
    exp_name = '%s-X-%s' % (model_name, data_name)

    fold_index = int(arguments['<fold_index>'])
    every_K_frame = int(arguments['<every_K_frame>'])
    plot_folder = '%s/%s' % (CONFIG.plots.dir, exp_name)

    make_raw(data_config, model_config, exp_name, fold_index, every_K_frame, plot_folder)
    detect_from_prob(data_config, model_config, detect_config, exp_name, fold_index, plot_folder)
