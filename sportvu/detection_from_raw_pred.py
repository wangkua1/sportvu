"""detection_from_raw_pred.py

Usage:
    detection_from_raw_pred.py <fold_index> <f_data_config> <f_model_config> <f_detect_config>

Arguments:
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
all_pred_f = filter(lambda s:'.pkl' in s,os.listdir(plot_folder))
for ind, f in tqdm(enumerate(all_pred_f)):
    gameclocks, pnr_probs, labels = pkl.load(open(os.path.join(plot_folder,'%i.pkl'%(ind)), 'rb'))
    cands = detector.detect(pnr_probs, gameclocks)
    print (cands)
    plt.plot(gameclocks, pnr_probs, '-')
    plt.plot(np.array(labels), np.ones((len(labels))), '.')
    for cand in cands:
        cand_x = np.arange(cand[1], cand[0], .1)
        plt.plot(cand_x, np.ones((len(cand_x))) * .95, '-' )
    plt.savefig(os.path.join(plot_folder, '%s-%i.png' %(detect_config['detect_config']['type'], ind)))
    plt.clf()