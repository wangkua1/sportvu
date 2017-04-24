
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os
# data
from sportvu.data.dataset import BaseDataset
from sportvu.data.extractor import EncDecExtractor, Seq2SeqExtractor
from sportvu.data.loader import Seq2SeqLoader
# concurrent
from tqdm import tqdm
from docopt import docopt
import yaml
import gc
import matplotlib.pylab as plt
from sportvu.data.utils import pictorialize_fast

def _pictorialize_single_sequence(seq, kwargs={'sample_rate':1, 'Y_RANGE':100, 'X_RANGE':50, 'keep_channels':False}):
    shape = list(seq.shape) #(B, T, 2)
    shape.insert(1,11)
    tmp = np.zeros((shape))
    tmp[:,0]= seq
    tmp_img = pictorialize_fast(tmp, **kwargs)
    tmp_img = np.sum(tmp_img[:,0],axis=1) # (B, X, Y)
    tmp_img[tmp_img>1] = 1
    return tmp_img

f_data_config = '../sportvu/data/config/rev3-ed-target-history.yaml'
data_config = yaml.load(open(f_data_config, 'rb'))
data_config['batch_size']  = 1
# Initialize dataset/loader
dataset = BaseDataset(data_config, 0, load_raw=True)
# extractor = Seq2SeqExtractor(data_config)
extractor = EncDecExtractor(data_config)
loader = Seq2SeqLoader(dataset, extractor, data_config[
    'batch_size'], fraction_positive=0.5)
### visualize sequence with unreasonably large step
N=10000
all_x = []
x = []
metas = []
for i in tqdm(xrange(N)):
    loaded = loader.next()
    if loaded is not None:
        dec_input, dec_output, enc_input, (meta)  = loaded
        all_x.append(dec_output)
        if np.max(np.abs(dec_output)) > 5:
            x.append(dec_output) 
            metas.append(meta)
        # _,_,_, dec_output = loaded
    else:
        loader.reset()
        continue
# # histogram
# plt.ion()
# n, bins, patches = plt.hist(np.abs(np.ravel(all_x)), 200, range=(0,50))
# plt.ylim(0,50)
# plt.xlim(0,50)
# plt.show()
# plt.title("SportVU dx/frame\nafter filtering out discontinuous sequences")
# plt.xlabel('ft/frame')
# plt.ylabel('frequency')
# plt.grid(b=True, which='both', color='0.65',linestyle='-')
# plt.plot([1.4,1.4],[0,50],color='r', label='Usain Bolt footspeed record (1.4 ft/fr)')
# plt.legend()

# from sportvu.utils import find_discontinuous_sequences
# ## visualize unreasonably large
# N = 10
# imgs = []
# for i in xrange(N):
#     history, pid, problematic_sequences = metas[i]
#     imgs.append(pictorialize_fast(problematic_sequences)[0])
# imgs = np.array(imgs)
# imgs = np.sum(imgs, axis=2) # (N, 3, 100,50)
# imgs = np.transpose(imgs, (0,2,3,1))
# show_img = np.concatenate(list(imgs), axis=1)

# ### filter discontinuous events
# N = 10
# imgs = []


# for i in xrange(N):
#     history, pid, problematic_sequences = metas[i]
#     ## (1,11,20,2)
#     if not find_discontinuous_sequences(problematic_sequences)[0]:
#         imgs.append(pictorialize_fast(problematic_sequences)[0])


# from sportvu.utils import truncated_mean, experpolate_position
# N = 10
# imgs = []
# for i in xrange(N):
#     history, pid = metas[i]
#     dec_output = x[i]
#     gt_future = experpolate_position(history[:,pid,-1], dec_output) #(1, T, 2)
#     # print (np.concatenate([gt_future[:,1:], real_future[:,pid,:-1]], axis=-1))
#     img = _pictorialize_single_sequence(gt_future) #(1,X,Y)
#     imgs.append(img)
# imgs= np.transpose(imgs, (0,2,3,1))[...,0] #(B, X,Y)
# show_img = np.concatenate(list(imgs), axis=-1) #(X, B*Y)



def print_mean_error(loader, N=100):
    l = []
    for i in tqdm(xrange(N)):
        loaded = loader.next()
        if loaded is not None:
            dec_input, dec_output, enc_input, (meta)  = loaded
            # _,_,_, dec_output = loaded
        else:
            loader.reset()
            continue
        ### mask out super huge noise (>50ft/s~=17m/s)
        dec_output[dec_output>2] = 0
        l.append(np.power(dec_output, 2).mean())
    print( np.mean(l) )
def print_linear_interpolation_error(loader, N=100):
    l = []
    for i in tqdm(xrange(N)):
        loaded = loader.next()
        if loaded is not None:
            dec_input, dec_output, enc_input, (meta)  = loaded
        else:
            loader.reset()
            continue
        ### mask out super huge noise (>50ft/s~=17m/s)
        bad_ind = dec_output>2
        dec_output[bad_ind] = 0
        dec_input[bad_ind] = 0
        bad_ind = dec_input>2
        dec_output[bad_ind] = 0
        dec_input[bad_ind] = 0
        

        l.append(np.power(dec_output - dec_input, 2).mean())
    print( np.mean(l) )
       