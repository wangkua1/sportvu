"""detection_pr.py
plot precision-recall, varying prob_threshold
Usage:
    detection_from_raw_pred.py <fold_index> <f_data_config> <f_model_config> <f_detect_config> <percent_grid>

Arguments:
    <percent_grid> e.g. 5, prob_threshold = 0,5,10,15 ...
Example:
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
##
from sportvu.detect.running_window_p import RunWindowP
from utils import smooth_1D_array
arguments = docopt(__doc__)
print ("...Docopt... ")
print(arguments)
print ("............\n")

f_data_config = arguments['<f_data_config>']
f_model_config = arguments['<f_model_config>']
f_detect_config = arguments['<f_detect_config>']
# pre_trained = arguments['<pre_trained>']
data_config = yaml.load(open(f_data_config, 'rb'))
model_config = yaml.load(open(f_model_config, 'rb'))
model_name = os.path.basename(f_model_config).split('.')[0]
data_name = os.path.basename(f_data_config).split('.')[0]
exp_name = '%s-X-%s' % (model_name, data_name)
detect_config = yaml.load(open(f_detect_config, 'rb'))

detector = RunWindowP(detect_config)


plot_folder = os.path.join('./plots', exp_name)
if not os.path.exists(plot_folder):
    raise Exception('Run test.py first to get raw predictions')


plt.figure()
all_pred_f = filter(lambda s: '.pkl' in s, os.listdir(plot_folder))


def PR(all_pred_f, detector):
    relevant = 0
    retrieved = 0
    intersect = 0
    for ind, f in tqdm(enumerate(all_pred_f)):
        gameclocks, pnr_probs, labels = pkl.load(
            open(os.path.join(plot_folder, '%i.pkl' % (ind)), 'rb'))
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
        return 0,0
    return intersect / retrieved, intersect / relevant


plt.figure()
for count_threshold in xrange(1,detector.config['window_radius']*2,2):
    detector.config['count_threshold'] = count_threshold
    ps =[]
    rs =[]
    for prob_threshold in tqdm(xrange(0,100,int(arguments['<percent_grid>']))):
        detector.config['prob_threshold'] = prob_threshold *.01
        precision, recall = PR(all_pred_f, detector)
        ps.append(precision)
        rs.append(recall)
    plt.plot(rs, ps, label='count-thresh:%i'%count_threshold)

plt.xlim([0,1])
plt.ylim([0,1])
plt.title('precision-recall')
plt.xlabel('recall')
plt.ylabel('precision')
plt.legend()
plt.savefig(os.path.join(plot_folder, 'precision-recall.png'))
plt.clf()