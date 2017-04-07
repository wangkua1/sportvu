"""make_sequence_data.py

This prepare a dataset after only extraction, not preprocessing
Th

Usage:
    make_sequence_data.py <f_data_config> 

Arguments:
    
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

f_data_config = 'data/config/train_rev0.yaml'
fold_index = 0
dataset = BaseDataset(f_data_config, fold_index=fold_index)
extractor = BaseExtractor(f_data_config)
loader = BaseLoader(dataset, extractor, 32, fraction_positive=.5)



batch = loader.next_batch(False)
sequences = np.array([make_3teams_11players(extractor.extract_raw(e)) for e in batch[0]])
model_x = pictorialize_fast(sequences)
# for i in tqdm(range(10)):
#     sequences = np.array([make_3teams_11players(extractor.extract_raw(e)) for e in batch[0]])
#     model_x = pictorialize_fast(sequences)
# for i in tqdm(range(10)):
#     for j in xrange(len(batch[0])):
#         a = extractor.extract(batch[0][j])




