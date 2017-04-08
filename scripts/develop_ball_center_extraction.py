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


f_data_config = '../sportvu/data/config/rev2-tb1.yaml'
fold_index = 0
dataset = BaseDataset(f_data_config, fold_index=fold_index, load_raw=False)
extractor = BaseExtractor(f_data_config)
loader = SequenceLoader(dataset, extractor, 32, fraction_positive=.5)

x, t = loader.next_batch(False)



crop_size = [11, 11]
reference = make_reference(x, crop_size,'tb')
x = pictorialize_fast(x - reference, 1, crop_size[0]+2, crop_size[1]+2)
x = x[:,:,:,1:-1,1:-1]


# batch = loader.next_batch(False)
# sequences = np.array([make_3teams_11players(extractor.extract_raw(e)) for e in batch[0]])
# model_x = pictorialize_fast(sequences)
# for i in tqdm(range(10)):
#     sequences = np.array([make_3teams_11players(extractor.extract_raw(e)) for e in batch[0]])
#     model_x = pictorialize_fast(sequences)
# for i in tqdm(range(10)):
#     for j in xrange(len(batch[0])):
#         a = extractor.extract(batch[0][j])


# N_iter = 10
# for i in tqdm(range(N_iter)):
#   batch = loader.next_batch(False)

# for i in tqdm(range(N_iter)):
#   [extractor.extract_raw(e) for e in batch[0]]

# for i in tqdm(range(N_iter)):
#   sequences = np.array([make_3teams_11players(extractor.extract_raw(e)) for e in batch[0]])

# for i in tqdm(range(N_iter)):
#   model_x = pictorialize_fast(sequences)
