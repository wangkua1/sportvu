"""make_hard_negatives.py

really the same functionality as make_sequences_from_sportvu.py
but it's used at different stages. Used to augment an existing dataset
with hard negative examples

Usage:
    make_hard_negatives.py <fold_index> <f_data_config> 

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

##
new_root = os.path.join(data_dir, data_config['preproc_dir'])
assert(os.path.exists(new_root))
# if not, make a dataset first using make_sequences_from_sportvu

# save the configuartion
with open(os.path.join(new_root, 'config.yaml'), 'w') as outfile:
    yaml.dump(data_config, outfile)

fold_index = int(arguments['<fold_index>'])
curr_folder = os.path.join(new_root, '%i' % fold_index)
assert(os.path.exists(curr_folder))
# Initialize dataset/loader
dataset = BaseDataset(f_data_config, fold_index=fold_index)
extractor = BaseExtractor(f_data_config)
loader = BaseLoader(dataset, extractor, data_config[
                    'n_negative_examples'], fraction_positive=0)

x = loader.load_by_annotations(dataset.hard_negatives, extract=False)
t = np.array([[1, 0]]).repeat(len(x), axis=0)
x = np.array([make_3teams_11players(extractor.extract_raw(e))
              for e in x])
np.save(os.path.join(curr_folder, 'hard_neg_x'), x)
np.save(os.path.join(curr_folder, 'hard_neg_t'), t)
