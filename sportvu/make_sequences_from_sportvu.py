"""make_sequences_from_sportvu.py

Usage:
    make_sequences_from_sportvu.py <f_data_config> 

Arguments:
    <f_data_config>  example ''data/config/rev2.yaml''

Example:
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
from sportvu.data.utils import make_3teams_11players
arguments = docopt(__doc__)
print ("...Docopt... ")
print(arguments)
print ("............\n")

f_data_config = arguments['<f_data_config>']
data_config = yaml.load(open(f_data_config, 'rb'))

# make a new data directions
if ('<new_data_dir>' in arguments and
        arguments['<new_data_dir>'] != None):
    assert (arguments['<new_data_dir>'] == data_config['preproc_dir'])

new_root = os.path.join(data_dir, data_config['preproc_dir'])
if not os.path.exists(new_root):
    os.makedirs(new_root)

# save the configuartion
with open(os.path.join(new_root, 'config.yaml'), 'w') as outfile:
    yaml.dump(data_config, outfile)


for fold_index in tqdm(xrange(data_config['data_config']['N_folds'])):
    curr_folder = os.path.join(new_root, '%i' % fold_index)
    if not os.path.exists(curr_folder):
        os.makedirs(curr_folder)
    # Initialize dataset/loader
    dataset = BaseDataset(f_data_config, fold_index=fold_index)
    extractor = BaseExtractor(f_data_config)
    loader = BaseLoader(dataset, extractor, data_config[
                        'n_negative_examples'], fraction_positive=0)

    vx, vt = loader.load_valid(False)
    vx = np.array([make_3teams_11players(extractor.extract_raw(e))
                   for e in vx])
    np.save(os.path.join(curr_folder, 'vx'), vx)
    np.save(os.path.join(curr_folder, 'vt'), vt)

    x, t = loader.load_train(extract=False, positive_only=True)
    x = np.array([make_3teams_11players(extractor.extract_raw(e))
                  for e in x])
    np.save(os.path.join(curr_folder, 'pos_x'), x)
    np.save(os.path.join(curr_folder, 'pos_t'), t)

    x, t = loader.next_batch(extract=False)
    x = np.array([make_3teams_11players(extractor.extract_raw(e))
                  for e in x])
    np.save(os.path.join(curr_folder, 'neg_x'), x)
    np.save(os.path.join(curr_folder, 'neg_t'), t)
