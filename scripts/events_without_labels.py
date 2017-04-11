from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
# data
from sportvu.data.dataset import BaseDataset
from sportvu.data.extractor import BaseExtractor
from sportvu.data.loader import BaseLoader, PreprocessedLoader, EventLoader, SequenceLoader
from tqdm import tqdm
from sportvu.data.utils import make_3teams_11players, pictorialize_fast, make_reference


f_data_config = '../sportvu/data/config/rev2.yaml'
fold_index = 0
dataset = BaseDataset(f_data_config, fold_index=fold_index, load_raw=True)
extractor = BaseExtractor(f_data_config)
loader = SequenceLoader(dataset, extractor, 32, fraction_positive=.5)


