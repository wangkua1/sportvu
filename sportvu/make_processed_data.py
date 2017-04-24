"""make_processed_data.py
OBSOLETE
Usage:
    make_one_game.py <f_data_config> <new_data_dir> <n_batches>
    make_one_game.py <f_data_config> 

Arguments:
    <f_data_config>  example ''data/config/train_rev0.yaml''
    <new_data_dir> whatever appended to sportvu.constant.data_dir (OBSOLETE, now in data_config)

Example:
    python make_processed_data.py data/config/train_rev0_circle.yaml rev0_circle 5000
"""
from sportvu.data.dataset import BaseDataset
from sportvu.data.extractor import BaseExtractor
from sportvu.data.loader import BaseLoader
from sportvu.data.constant import data_dir
from tqdm import tqdm
import os
from docopt import docopt
import yaml
import numpy as np

arguments = docopt(__doc__)
print ("...Docopt... ")
print(arguments)
print ("............\n")

f_data_config = arguments['<f_data_config>']
data_config = yaml.load(open(f_data_config, 'rb'))

## make a new data directions
if arguments['<new_data_dir>'] != None:
    assert (arguments['<new_data_dir>'] == data_config['preproc_dir'])

new_root = os.path.join(data_dir, data_config['preproc_dir'])
if not os.path.exists(new_root):
    os.makedirs(new_root)

## save the configuartion 
with open(os.path.join(new_root, 'config.yaml'), 'w') as outfile:
    yaml.dump(data_config, outfile)


for fold_index in xrange(data_config['data_config']['N_folds']):
    curr_folder = os.path.join(new_root, '%i'%fold_index)
    if not os.path.exists(curr_folder):
        os.makedirs(curr_folder)
    # Initialize dataset/loader
    dataset = BaseDataset(f_data_config, fold_index=fold_index)
    extractor = BaseExtractor(f_data_config)
    loader = BaseLoader(dataset, extractor, 32, fraction_positive=.5)

    if 'no_extract' in data_config and data_config['no_extract']:
        vx, vt = loader.load_valid(False)
    else:
        vx, vt = loader.load_valid()
    np.save(os.path.join(curr_folder, 'vx'), vx)
    np.save(os.path.join(curr_folder, 'vt'), vt)

    for batch_index  in tqdm(xrange(int(data_config['n_batches']))):
        if 'no_extract' in data_config and  data_config['no_extract']:
            x, t = loader.next_batch(False)
        else:
            x, t = loader.next()
        np.save(os.path.join(curr_folder, '%ix'%batch_index), x)
        np.save(os.path.join(curr_folder, '%it'%batch_index), t)




