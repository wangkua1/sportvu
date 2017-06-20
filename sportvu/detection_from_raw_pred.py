"""detection_from_raw_pred.py
* not super useful, a simple script that plots a) raw pred, b) gt pnr, c) detector output
at 1 single setting
Usage:
    detection_from_raw_pred.py <fold_index> <f_data_config> <f_model_config> <f_detect_config> --train

Arguments:
Example: python detection_from_raw_pred.py 1 rev3_1-bmf-25x25.yaml conv2d-3layers-25x25.yaml nms1.yaml --train
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
from sportvu.data.dataset import BaseDataset
from sportvu.detect.running_window_p import RunWindowP
from sportvu.detect.nms import NMS
from sportvu.detect.utils import smooth_1D_array
# configuration
import config as CONFIG

arguments = docopt(__doc__)
print ("...Docopt... ")
print(arguments)
print ("............\n")

f_data_config = CONFIG.data.config.dir + '/' + arguments['<f_data_config>']
f_model_config = CONFIG.model.config.dir + '/' + arguments['<f_model_config>']
f_detect_config = CONFIG.detect.config.dir + '/' + arguments['<f_detect_config>']
if arguments['--train']:
    dataset = BaseDataset(f_data_config, fold_index=int(arguments['<fold_index>']), load_raw=True)
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

def label_in_cand(cand, labels):
    for l in labels:
        if l > cand[1] and l < cand[0]:
            return True
    return False

plt.figure()
if arguments['--train']:
    split = 'train'
else:
    split = 'val'
all_pred_f = filter(lambda s:'.pkl' in s and split in s
                    and 'meta' not in s,os.listdir('%s/pkl'%(plot_folder))
if arguments['--train']:
    annotations = []
for _, f in tqdm(enumerate(all_pred_f)):
    ind = int(f.split('.')[0].split('-')[1])
    gameclocks, pnr_probs, labels = pkl.load(open('%s/pkl/%s-%i.pkl'%(plot_folder,split,ind), 'rb'))
    meta = pkl.load(open('%s/pkl/%s-meta-%i.pkl' %(plot_folder,split, ind), 'rb'))
    cands, mp, frame_indices = detector.detect(pnr_probs, gameclocks, True)
    print (cands)
    plt.plot(gameclocks, pnr_probs, '-')
    if mp is not None:
        plt.plot(gameclocks, mp, '-')
    plt.plot(np.array(labels), np.ones((len(labels))), '.')
    for ind, cand in enumerate(cands):
        cand_x = np.arange(cand[1], cand[0], .1)
        plt.plot(cand_x, np.ones((len(cand_x))) * .95, '-' )
        ## if FP, record annotations
        if arguments['--train'] and not label_in_cand(cand, labels):
            anno = {'gameid':meta[1], 'gameclock':gameclocks[frame_indices[ind]],
                    'eid':meta[0], 'quarter':dataset.games[meta[1]]['events'][meta[0]]['quarter']}
            annotations.append(anno)
    plt.ylim([0,1])
    plt.title('Game: %s, Event: %i'%(meta[1], meta[0]))
    plt.savefig('%s/%s-%s-%i.png' %(plot_folder,detect_config['class'], split, ind))
    plt.clf()
pkl.dump(annotations, open('%s/pkl/hard-negative-examples.pkl'%(plot_folder), 'wb'))
