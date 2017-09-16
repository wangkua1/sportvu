"""detection_pr.py
plot precision-recall, varying prob_threshold
Usage:
    detection_pr.py <fold_index> <f_data_config> <f_model_config> <f_detect_config> <grid_type> <percent_grid> --train
    detection_pr.py <fold_index> <f_data_config> <f_model_config> <f_detect_config> <grid_type> <percent_grid> --test
    detection_pr.py <fold_index> <f_data_config> <f_model_config> <f_detect_config> <grid_type> <percent_grid>

Arguments:
    <percent_grid> e.g. 5, prob_threshold = 0,5,10,15 ...
Example:
    python detection_pr.py 0 rev3_1-bmf-25x25.yaml conv2d-3layers-25x25.yaml nms1.yaml single 5
    python detection_pr.py 0 rev3_1-bmf-25x25.yaml conv2d-3layers-25x25.yaml window-5-thresh-80-fixed.yaml auc 5
    python detection_pr.py 0 rev3_1-bmf-25x25.yaml conv2d-3layers-25x25.yaml window-5-thresh-80-fixed.yaml many 5
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import sys
import os
from tqdm import tqdm
from docopt import docopt
import yaml
import gc
import matplotlib.pylab as plt
import cPickle as pkl
from sklearn import metrics
##
from sportvu.detect.running_window_p import RunWindowP
from sportvu.detect.nms import NMS
# configuration
import config as CONFIG

arguments = docopt(__doc__)
print ("...Docopt... ")
print(arguments)
print ("............\n")

f_data_config = '%s/%s' % (CONFIG.data.config.dir,arguments['<f_data_config>'])
f_model_config = '%s/%s' % (CONFIG.model.config.dir,arguments['<f_model_config>'])
f_detect_config = '%s/%s' % (CONFIG.detect.config.dir,arguments['<f_detect_config>'])
# pre_trained = arguments['<pre_trained>']
data_config = yaml.load(open(f_data_config, 'rb'))
model_config = yaml.load(open(f_model_config, 'rb'))
model_name = os.path.basename(f_model_config).split('.')[0]
data_name = os.path.basename(f_data_config).split('.')[0]
exp_name = '%s-X-%s' % (model_name, data_name)
detect_config = yaml.load(open(f_detect_config, 'rb'))

detector = eval(detect_config['class'])(detect_config)


plot_folder = '%s/%s' % (CONFIG.plots.dir,exp_name)
if not os.path.exists(plot_folder):
    raise Exception('Run test.py first to get raw predictions')


plt.figure()
if arguments['--train']:
    split = 'train'
    all_pred_f = filter(lambda s: '.pkl' in s and split in s
                                  and 'meta' not in s, os.listdir('%s/pkl' % (plot_folder)))
elif arguments['--test']:
    split = 'val'
    all_pred_f = filter(lambda s: '.pkl' in s and split in s
                                  and 'meta' not in s, os.listdir('%s/pkl' % (plot_folder)))
else:
    split = 'raw'
    all_pred_f = filter(lambda s: 'raw-' in s and 'raw-meta' not in s, os.listdir('%s/pkl' % (plot_folder)))


def PR(all_pred_f, detector):
    relevant = 0
    retrieved = 0
    intersect = 0
    for ind, f in tqdm(enumerate(all_pred_f)):
        gameclocks, pnr_probs, labels = pkl.load(open('%s/pkl/%s' % (plot_folder, f), 'rb'))
        cands = detector.detect(pnr_probs, gameclocks)
        for label in labels:
            label_detected = False
            for cand in cands:
                if label > cand[1] and label < cand[0]:
                    label_detected = True
                    break
            if label_detected:
                intersect += 1
        relevant += len(labels)
        retrieved += len(cands)
    if intersect == 0:
        return 0, 0
    return intersect / retrieved, intersect / relevant

fig = plt.figure()


if detect_config['class'] == 'RunWindowP':
    # outer_grid = xrange(1, detector.config['window_radius'] * 2, 2)
    outer_grid = xrange(5, 15, 1)
    k = 'count_threshold'
elif detect_config['class'] == 'NMS':
    outer_grid = xrange(1, detector.config['instance_radius'], 2)
    k = 'instance_radius'
if arguments['<grid_type>'] == 'single':
    out_type = 'single'
    ps = []
    rs = []
    for prob_threshold in tqdm(xrange(0, 100, int(arguments['<percent_grid>']))):
        detector.config['prob_threshold'] = prob_threshold * .01
        print('Probability %f' % (prob_threshold * .01))
        precision, recall = PR(all_pred_f, detector)
        ps.append(precision)
        rs.append(recall)
    plt.plot(np.arange(0, 1, .01*int(
        arguments['<percent_grid>'])), ps, label='precision')
    plt.plot(np.arange(0, 1, .01*int(
        arguments['<percent_grid>'])), rs, label='recall')
    plt.xlabel('prob_threshold')
    plt.ylabel('precision/recall')
elif arguments['<grid_type>'] == 'auc' or arguments['<f_data_config>'] == 'AUC':
    out_type = 'auc'
    pr = []
    rc = []
    for prob_threshold in tqdm(xrange(0, 100, int(arguments['<percent_grid>']))):
        detector.config['prob_threshold'] = prob_threshold * .01
        print('Probability %f' % (prob_threshold * .01))
        precision, recall = PR(all_pred_f, detector)
        pr.append(precision)
        rc.append(recall)
    auc = metrics.auc(rc, pr)
    plt.plot(rc, pr, label='AUC: %f' % (auc))
    # plt.plot(rc, pr)
    plt.xlabel('recall')
    plt.ylabel('precision')
else:
    out_type = 'many'
    for v in outer_grid:
        detector.config[k] = v
        ps = []
        rs = []
        for prob_threshold in tqdm(xrange(0, 100,
                                          int(arguments['<percent_grid>']))):
            detector.config['prob_threshold'] = prob_threshold * .01
            precision, recall = PR(all_pred_f, detector)
            ps.append(precision)
            rs.append(recall)
        plt.plot(rs, ps, label='%s:%i' % (k, v))
    plt.xlabel('recall')
    plt.ylabel('precision')

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title('precision-recall')
plt.legend()
ax = fig.gca()
ax.set_xticks(np.arange(0, 1, 0.1))
ax.set_yticks(np.arange(0, 1., 0.1))
plt.grid()
plt.savefig('%s/%s-precision-recall-%s.png'%(CONFIG.plots.dir,detect_config['class'], out_type))
plt.clf()
