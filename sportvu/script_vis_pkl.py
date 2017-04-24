"""script_vis_pkl.py

visualize the pkls in <dir>

Usage:
    script_vis_pkl.py <dir>
"""
from utils import truncated_mean, experpolate_position
from vis_utils import make_sequence_prediction_image
import cPickle as pkl
import os
import matplotlib.pylab as plt
from docopt import docopt
from tqdm import tqdm

plt.ioff()
fig = plt.figure()


arguments = docopt(__doc__)
print ("...Docopt... ")
print(arguments)
print ("............\n")
target_dir = arguments['<dir>']

pkls = filter(lambda s:'.pkl' in s, os.listdir(target_dir))
for f_pkl in tqdm(pkls):
    history, gt_future, pred_future, pid = pkl.load(open(os.path.join(target_dir, f_pkl),'rb'))
    imgs = make_sequence_prediction_image(history, gt_future, pred_future, pid)

    for i in xrange(5):
        plt.imshow(imgs[i])
        plt.savefig(os.path.join(target_dir, '%s-%g.png'%(f_pkl.split('.')[0],i)))
    