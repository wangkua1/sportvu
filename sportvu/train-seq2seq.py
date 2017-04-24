"""train-seq2seq.py

Usage:
    train-seq2seq.py <fold_index> <f_data_config> <f_model_config>
    train-seq2seq.py --test <fold_index> <f_data_config> <f_model_config>

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
from vis_utils import make_sequence_prediction_image
import cPickle as pkl
# import matplotlib.pylab as plt
# plt.ioff()
# fig = plt.figure()

def train(data_config, model_config, exp_name, fold_index, init_lr, max_iter, best_acc_delay, testing=False):
    # Initialize dataset/loader
    dataset = BaseDataset(data_config, fold_index, load_raw=True)
    extractor = eval(data_config['extractor_class'])(data_config)
    if 'negative_fraction_hard' in data_config:
        nfh = data_config['negative_fraction_hard']
    else:
        nfh = 0

    loader = Seq2SeqLoader(dataset, extractor, data_config[
        'batch_size'], fraction_positive=0.5,
        negative_fraction_hard=nfh)
    Q_size = 100
    N_thread = 4
    # cloader = ConcurrentBatchIterator(
    #     loader, max_queue_size=Q_size, num_threads=N_thread)
    cloader = loader

    net = eval(model_config['class_name'])(model_config['model_config'])
    net.build()

    # build loss
    y_ = tf.placeholder(tf.float32,
                        [model_config['model_config']['batch_size'],
                         model_config['model_config']['decoder_time_size'],
                         2])
    learning_rate = tf.placeholder(tf.float32, [])
    # euclid_loss = tf.reduce_mean(tf.pow(net.output() - y_, 2))
    euclid_loss = tf.reduce_mean(tf.pow(tf.reduce_sum(tf.pow(net.output() - y_, 2), axis=-1),.5))

    global_step = tf.Variable(0)
    # train_step = optimize_loss(euclid_loss, global_step, learning_rate,
    #                            optimizer=lambda lr: tf.train.AdamOptimizer(lr),
    #                            clip_gradients=0.01)
    train_step = optimize_loss(euclid_loss, global_step, learning_rate,
                               optimizer=lambda lr: tf.train.RMSPropOptimizer(lr),
                               clip_gradients=0.01)
    # train_step = optimize_loss(cross_entropy, global_step, learning_rate,
    #             optimizer=lambda lr: tf.train.MomentumOptimizer(lr, .9))

    # # testing
    # if testing:
    #     saver = tf.train.Saver()
    #     sess = tf.InteractiveSession()
    #     ckpt_path = os.path.join("./saves/", exp_name + '.ckpt.best')
    #     saver.restore(sess, ckpt_path)

    #     feed_dict = net.input(val_x, 1, False)
    #     feed_dict[y_] = val_t
    #     ce, val_accuracy = sess.run([cross_entropy, accuracy], feed_dict=feed_dict)
    #     print ('Best Validation CE: %f, Acc: %f' % (ce, val_accuracy))
    #     sys.exit(0)

    # checkpoints
    if not os.path.exists('./saves'):
        os.mkdir('./saves')
    # tensorboard
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    v_loss = tf.Variable(tf.constant(0.0), trainable=False)
    v_loss_pl = tf.placeholder(tf.float32, shape=[], name='v_loss_pl')
    update_v_loss = tf.assign(v_loss, v_loss_pl, name='update_v_loss')
    v_rloss = tf.Variable(tf.constant(0.0), trainable=False)
    v_rloss_pl = tf.placeholder(tf.float32, shape=[])
    update_v_rloss = tf.assign(v_rloss, v_rloss_pl)
    tf.summary.scalar('euclid_loss', euclid_loss)
    tf.summary.scalar('valid_loss', v_loss)
    # tf.summary.image('encoder_input', tf.transpose(
    #     tf.reduce_sum(net.tf_enc_input, 2), (0, 2, 3, 1))[:,:,:,:-1], max_outputs=4)
    # tf.summary.image('decoder_input', tf.transpose(
    #     tf.reduce_sum(net.tf_dec_input, 2), (0, 2, 3, 1))[:,:,:,:-1], max_outputs=4)
    # TODO
    # tf.summary.image('prediction', tf.reduce_sum(net.x, 1), max_outputs=4)
    # tf.summary.image('groundtruth', tf.reduce_sum(net.x, 1), max_outputs=4)

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
    val_writer = tf.summary.FileWriter(
        os.path.join(log_folder, 'val'), sess.graph)
    tf.global_variables_initializer().run()
    # Train
    best_val_teacher_forced_loss = np.inf
    best_val_real_loss = np.inf
    best_not_updated = 0
    lrv = init_lr
    tfs = model_config['model_config']['decoder_time_size']
    train_loss = []
    for iter_ind in tqdm(range(max_iter)):
        best_not_updated += 1
        loaded = cloader.next()
        if loaded is not None:
            dec_input, dec_output, enc_input,_= loaded
        else:
            cloader.reset()
            continue
        if iter_ind>0 and iter_ind % 5000 == 0:
            tfs -= 5
        feed_dict = net.input(dec_input, 
                                teacher_forcing_stop=np.max([1, tfs]),
                                enc_input=enc_input
                                )
        feed_dict[y_] = dec_output
        if iter_ind ==2000:
            lrv *= .1
        feed_dict[learning_rate] = lrv
        summary, tl = sess.run([merged, train_step], feed_dict=feed_dict)
        # print (tl)
        train_loss.append(tl)
        train_writer.add_summary(summary, iter_ind)
        # Validate
        if iter_ind % 1000 == 0:
            val_tf_loss = []
            val_real_loss = []
            ## loop throught valid examples
            while True:
                loaded = cloader.load_valid()
                if loaded is not None:
                    dec_input, dec_output, enc_input, (meta)  = loaded
                    history, pid = meta
                else: ## done
                    # print ('...')
                    break
                ## teacher-forced loss
                feed_dict = net.input(dec_input, 
                                teacher_forcing_stop=None,
                                enc_input=enc_input,
                                enc_keep_prob = 1.
                                )
                feed_dict[y_] = dec_output
                val_loss = sess.run(euclid_loss, feed_dict = feed_dict)
                val_tf_loss.append(val_loss)
                ## real-loss
                feed_dict = net.input(dec_input, 
                                teacher_forcing_stop=1,
                                enc_input=enc_input,
                                enc_keep_prob = 1.
                                )
                feed_dict[y_] = dec_output
                val_loss = sess.run(euclid_loss, feed_dict = feed_dict)
                val_real_loss.append(val_loss)
                ### plot
                pred = sess.run(net.output(), feed_dict = feed_dict)
                gt_future = experpolate_position(history[:,pid,-1], dec_output)
                pred_future = experpolate_position(history[:,pid,-1], pred)
                # imgs = make_sequence_prediction_image(history, gt_future, pred_future, pid)
                pkl.dump((history, gt_future, pred_future, pid),
                          open(os.path.join("./logs/"+exp_name, 'iter-%g.pkl'%(iter_ind)),'wb'))

                # for i in xrange(5):
                #     plt.imshow(imgs[i])
                #     plt.savefig(os.path.join("./saves/", exp_name +'iter-%g-%g.png'%(iter_ind,i)))
                
                
            ## TODO: evaluate real-loss on training set
            val_tf_loss = np.mean(val_tf_loss)
            val_real_loss = np.mean(val_real_loss)
            print ('[Iter: %g] Train Loss: %g, Validation TF Loss: %g | Real Loss: %g ' %(iter_ind,truncated_mean(train_loss),val_tf_loss, val_real_loss))
            train_loss = []
            feed_dict[v_loss_pl] = val_tf_loss
            feed_dict[v_rloss_pl] = val_real_loss
            _,_, summary = sess.run([update_v_loss,update_v_rloss, merged], feed_dict=feed_dict)
            val_writer.add_summary(summary, iter_ind)
            # if val_ce < best_val_ce:
            #     best_not_updated = 0
            #     p = os.path.join("./saves/", exp_name + '.ckpt.best')
            #     print ('Saving Best Model to: %s' % p)
            #     save_path = saver.save(sess, p)
            #     best_val_ce = val_ce
        if iter_ind % 2000 == 0:
            save_path = saver.save(sess, os.path.join(
                "./saves/", exp_name + '%d.ckpt' % iter_ind))
        # if best_not_updated == best_acc_delay:
        #     break
    return _


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
    init_lr = 1e-4
    max_iter = 100000
    best_acc_delay = 3000
    testing = arguments['--test']
    train(data_config, model_config, exp_name, fold_index,
          init_lr, max_iter, best_acc_delay, testing)
