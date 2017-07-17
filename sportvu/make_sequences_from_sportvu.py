"""make_sequences_from_sportvu.py

Usage:
    make_sequences_from_sportvu.py <f_data_config> --sample
    make_sequences_from_sportvu.py <f_data_config>

Arguments:
    <f_data_config>  example ''rev3_1-bmf-25x25.yaml''

Example:
    python make_sequences_from_sportvu.py rev3_1-bmf-25x25.yaml
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
# configuration
import config as CONFIG

arguments = docopt(__doc__)
print ("...Docopt... ")
print(arguments)
print ("............\n")

f_data_config = '%s/%s'%(CONFIG.data.config.dir,arguments['<f_data_config>'])
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


# for fold_index in tqdm(xrange(data_config['data_config']['N_folds'])):
for fold_index in tqdm(xrange(1)): ## I have never actually used more than 1 fold...
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

    del vx, vt

    x, t = loader.load_train(extract=False, positive_only=True)
    x = np.array([make_3teams_11players(extractor.extract_raw(e))
                  for e in x])
    np.save(os.path.join(curr_folder, 'pos_x'), x)
    np.save(os.path.join(curr_folder, 'pos_t'), t)

    del x

    if arguments['--sample']:
        x, t = loader.next_batch(extract=False)
    else:
        xs = []
        ind = 0
        while True:
            print ('%i/%i' % (ind, len(dataset.train_hash)))
            print (len(xs))
            ind+=1
            loaded = loader.load_split_event('train',extract=False)
            if loaded is not None:
                if loaded == 0:
                    continue
                batch_xs, labels, gameclocks, meta = loaded
                ## filter out positive examples
                new_batch_xs = []
                for x in batch_xs:
                    e_gc = x.moments[len(x.moments)/2].game_clock
                    ispositive = False
                    for label in labels:
                        if np.abs(e_gc-label) < data_config['data_config']['t_negative']:
                            ispositive = True
                            break
                    if not ispositive:
                        xs.append(x)
                if ind % 100 == 0 and ind > 100:
                    print('Saving split')
                    # npy file was written from previous split
                    x = xs
                    xs = []
                    x = np.array([make_3teams_11players(extractor.extract_raw(e))
                                  for e in x])
                    x_arr = np.load(os.path.join(curr_folder, 'neg_x.npy'))
                    x_arr = np.append(x_arr,x,axis=0)
                    np.save(os.path.join(curr_folder, 'neg_x'), x_arr)
                    del x_arr, x
                elif ind % 100 == 0 and ind == 100:
                    print('Saving split')
                    # no split has been written to file
                    x = xs
                    xs = []
                    x = np.array([make_3teams_11players(extractor.extract_raw(e))
                                  for e in x])
                    np.save(os.path.join(curr_folder, 'neg_x'), x)
                    del x
            else:
                print('Saving split')
                # last split
                x = xs
                x = np.array([make_3teams_11players(extractor.extract_raw(e))
                              for e in x])
                x_arr = np.load(os.path.join(curr_folder, 'neg_x.npy'))
                x_arr = np.append(x_arr,x,axis=0)
                np.save(os.path.join(curr_folder, 'neg_x'), x_arr)
                del x_arr, xs, x
                break

    np.save(os.path.join(curr_folder, 'neg_t'), t)
