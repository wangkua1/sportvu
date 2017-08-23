"""train-seq2seq.py

Usage:
    train-seq2seq.py <fold_index> <f_data_config> <f_model_config> <loss>
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
elif os.environ['HOME'] =='/u/ethanf':
    sys.path.append('/ais/gobi5/ethanf/projects/sportvu/sportvu/resnet')
    sys.path.append('/ais/gobi5/ethanf/projects/sportvu/sportvu')
    sys.path.append('/ais/gobi5/ethanf/projects/sportvu/')
    print(sys.path)
else:
    sys.path.append('/ais/gobi4/slwang/sports/sportvu/resnet')
    sys.path.append('/ais/gobi4/slwang/sports/sportvu')
from sportvu.model.seq2seq import Seq2Seq
from sportvu.model.encdec import EncDec, Velocity, Location
# data
from sportvu.data.dataset import BaseDataset
from sportvu.data.extractor import Seq2SeqExtractor, EncDecExtractor
from sportvu.data.loader import Seq2SeqLoader
from sportvu.data.utils import wrapper_concatenated_last_dim, scale_last_dim
# loss
from loss import RMSEPerPlayer, DiagGaussNLL, FullGaussNLL
# concurrent
# from resnet.utils.concurrent_batch_iter import ConcurrentBatchIterator
from tqdm import tqdm
from docopt import docopt
import yaml
import gc
from utils import truncated_mean, experpolate_position,dist_trajectory
from vis_utils import make_sequence_prediction_image
import cPickle as pkl
# import matplotlib.pylab as plt
# plt.ioff()
# fig = plt.figure()

def train(data_config, model_config, exp_name, fold_index, init_lr, max_iter, best_acc_delay, loss_class, testing=False):
    # Initialize dataset/loader
    dataset = BaseDataset(data_config, fold_index, load_raw=False)
    extractor = eval(data_config['extractor_class'])(data_config)
    if 'negative_fraction_hard' in data_config:
        nfh = data_config['negative_fraction_hard']
    else:
        nfh = 0

    loader = Seq2SeqLoader(dataset, extractor, data_config[
        'batch_size'], fraction_positive=0.5,
        negative_fraction_hard=nfh, move_N_neg_to_val=1000)
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
                         model_config['model_config']['dec_input_dim']])
    learning_rate = tf.placeholder(tf.float32, [])

    loss = loss_class.build_tf_loss(net.output(), y_)
    

    global_step = tf.Variable(0)
    # train_step = optimize_loss(loss, global_step, learning_rate,
    #                            optimizer=lambda lr: tf.train.AdamOptimizer(lr),
    #                            clip_gradients=0.01)
    train_step = optimize_loss(loss, global_step, learning_rate,
                               optimizer=lambda lr: tf.train.RMSPropOptimizer(lr),
                               clip_gradients=0.01)
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
    v_tloss = tf.Variable(tf.constant(0.0), trainable=False)
    v_tloss_pl = tf.placeholder(tf.float32, shape=[])
    update_v_tloss = tf.assign(v_tloss, v_tloss_pl)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('valid_loss', v_loss)
    tf.summary.scalar('real_valid_loss', v_rloss)
    tf.summary.scalar('real_traj_loss', v_tloss)
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
    best_saver = tf.train.Saver()
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
            val_traj_loss = []
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
                                enc_keep_prob = 1.,
                                decoder_noise_level = 0.,
                                decoder_input_keep_prob = 1.
                                )
                feed_dict[y_] = dec_output
                val_loss = sess.run(loss, feed_dict = feed_dict)
                val_tf_loss.append(val_loss)
                ## real-loss
                feed_dict = net.input(dec_input, 
                                teacher_forcing_stop=1,
                                enc_input=enc_input,
                                enc_keep_prob = 1.,
                                decoder_noise_level = 0.,
                                decoder_input_keep_prob = 1.
                                )
                feed_dict[y_] = dec_output
                val_loss = sess.run(loss, feed_dict = feed_dict)
                val_real_loss.append(val_loss)
                ### plot
                pred = sess.run(net.sample(), feed_dict = feed_dict)
                if pid is not None:
                    start_frame = history[:,pid, -1]
                else:
                    start_frame = history[:,:,-1].reshape(history.shape[0],-1)

                gt_future = experpolate_position(start_frame, dec_output)
                pred_future = experpolate_position(start_frame, pred)

                # imgs = make_sequence_prediction_image(history, gt_future, pred_future, pid)
                pkl.dump((history, gt_future, pred_future, pid),
                          open(os.path.join("./logs/"+exp_name, 'iter-%g.pkl'%(iter_ind)),'wb'))

                # for i in xrange(5):
                #     plt.imshow(imgs[i])
                #     plt.savefig(os.path.join("./saves/", exp_name +'iter-%g-%g.png'%(iter_ind,i)))
                

                ## TODO: always monitor loss using trajectory
                net.aux_feed_dict({'start_frame':start_frame}, feed_dict)
                traj = sess.run(net.sample_trajectory(), feed_dict = feed_dict)
                if model_config['class_name'] == 'Location': ### scale 
                    traj = wrapper_concatenated_last_dim(scale_last_dim, traj, upscale=True)
                    gt_traj = wrapper_concatenated_last_dim(scale_last_dim, dec_output, upscale=True)
                elif model_config['class_name'] == 'Velocity':
                    gt_traj = gt_future[:,1:]
                else:
                    raise NotImplementedError
                l_traj = dist_trajectory(traj, gt_traj)
                val_traj_loss.append(l_traj)
            ## TODO: evaluate real-loss on training set
            val_tf_loss = np.mean(val_tf_loss)
            val_real_loss = np.mean(val_real_loss)
            val_traj_loss = np.mean(val_traj_loss)
            print ('[Iter: %g] Train Loss: %g, Validation TF Loss: %g | Real Loss: %g | Real Traj Loss: %g' %(iter_ind,np.mean(train_loss),val_tf_loss, val_real_loss, val_traj_loss))
            train_loss = []
            feed_dict[v_loss_pl] = val_tf_loss
            feed_dict[v_rloss_pl] = val_real_loss
            feed_dict[v_tloss_pl] = val_traj_loss
            _,_,_, summary = sess.run([update_v_loss,update_v_rloss,update_v_tloss, merged], feed_dict=feed_dict)
            val_writer.add_summary(summary, iter_ind)
            if val_real_loss < best_val_real_loss:
                best_not_updated = 0
                p = os.path.join("./saves/", exp_name + '.ckpt.best')
                print ('Saving Best Model to: %s' % p)
                save_path = best_saver.save(sess, p)
                best_val_real_loss = val_real_loss
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
    exp_name = '%s-X-%s-X-%s' % (model_name, data_name,arguments['<loss>'])
    fold_index = int(arguments['<fold_index>'])
    init_lr = 1e-4
    max_iter = 100000
    best_acc_delay = 3000
    testing = arguments['--test']
    train(data_config, model_config, exp_name, fold_index,
          init_lr, max_iter, best_acc_delay, eval(arguments['<loss>'])(),testing)
