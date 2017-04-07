from __future__ import division
import cPickle as pickle
import yaml
import os
from sportvu import data
from sportvu.vis.Event import Event, EventException
from sportvu.data.extractor import ExtractorException
import numpy as np
game_dir = data.constant.game_dir
data_dir = data.constant.data_dir

class BaseLoader:
    def __init__(self, dataset, extractor, batch_size, mode='sample', fraction_positive=.5):
        self.dataset = dataset
        self.extractor = extractor
        self.batch_size = batch_size
        self.fraction_positive = fraction_positive
        self.mode = mode
        

    def next(self):
        """
        """
        if self.mode == 'sample':
            return self.next_batch()
        elif self.mode == 'valid':
            return self.load_valid()
        else:
            raise Exception('unknown loader mode')



    def next_batch(self, extract=True):
        N_pos = int(self.fraction_positive * self.batch_size)
        N_neg = self.batch_size - N_pos
        ret_val = []
        func = [self.dataset.propose_positive_Ta,
                self.dataset.propose_negative_Ta]
        Ns = [N_pos, N_neg]
        # anno = func[0]()
        for j in xrange(len(Ns)):
            for i in xrange(Ns[j]):
                while True:
                    try:
                        anno = func[j]()
                        e = Event(self.dataset.games[anno['gameid']][
                                  'events'][anno['eid']], gameid=anno['gameid'])
                        e.sequence_around_t(
                            anno['gameclock'], self.dataset.tfr)  # EventException
                        if extract:
                            # ExtractorException
                            ret_val.append(self.extractor.extract(e))
                        else:
                            _ = self.extractor.extract_raw(e) ## just to make sure event not malformed (like missing player)
                            ret_val.append(e)
                    except EventException as exc:
                        pass
                    except ExtractorException as exc:
                        pass
                    else:
                        break

        return (np.array(ret_val),
                np.vstack([np.array([[0, 1]]).repeat(N_pos, axis=0),
                           np.array([[1, 0]]).repeat(N_neg, axis=0)])
                )

    def load_valid(self, extract=True):
        N_pos = 0
        ret_val = []
        ret_labels = []
        func = [self.dataset.propose_positive_Ta,
                self.dataset.propose_negative_Ta]
        self.extractor.augment = False
        while True:
            anno = self.dataset.propose_positive_Ta(jitter=False, train=False)
            if anno == None:
                break
            try:
                e = Event(self.dataset.games[anno['gameid']][
                          'events'][anno['eid']], gameid=anno['gameid'])
                e.sequence_around_t(
                    anno['gameclock'], self.dataset.tfr)  # EventException
                if extract:
                    # ExtractorException
                    ret_val.append(self.extractor.extract(e))
                else:
                    ret_val.append(e)
            except EventException as exc:
                continue
            except ExtractorException as exc:
                continue
            else:
                N_pos += 1
                ret_labels.append([0,1])
        for i in xrange(N_pos):
            while True:
                try:
                    anno = self.dataset.propose_negative_Ta()
                    e = Event(self.dataset.games[anno['gameid']][
                              'events'][anno['eid']], gameid=anno['gameid'])
                    e.sequence_around_t(
                        anno['gameclock'], self.dataset.tfr)  # EventException
                    if extract:
                        # ExtractorException
                        ret_val.append(self.extractor.extract(e))
                    else:
                        ret_val.append(e)
                except EventException as exc:
                    pass
                except ExtractorException as exc:
                    pass
                else:
                    ret_labels.append([1,0])
                    break
        self.extractor.augment = True
        return (np.array(ret_val),
                np.array(ret_labels)
                )

    def reset(self):
        pass



class PreprocessedLoader:
    def __init__(self, dataset, extractor, batch_size, mode='sample', fraction_positive=.5):
        """ 
        simply loads numpy matrices from disk without preprocessing
        """
        self.dataset = dataset # not used
        self.root_dir = os.path.join(os.path.join(data_dir, self.dataset.config['preproc_dir']), str(self.dataset.fold_index))
        self.extractor = extractor # not used
        self.batch_size = batch_size # not used
        self.fraction_positive = fraction_positive
        self.mode = mode
        self.batch_index = 0
        self.dataset_size = self.dataset.config['n_batches']

    def next(self):
        """
        """
        if self.mode == 'sample':
            return self.next_batch()
        elif self.mode == 'valid':
            return self.load_valid()
        else:
            raise Exception('unknown loader mode')



    def next_batch(self, extract=True):
        if self.batch_index==self.dataset_size:
            return None
        x = np.load(os.path.join(self.root_dir , '%ix.npy'%self.batch_index))
        t = np.load(os.path.join(self.root_dir , '%it.npy'%self.batch_index))
        self.batch_index += 1
        return x, t

    def load_valid(self, extract=True):
        x = np.load(os.path.join(self.root_dir , 'vx.npy'))
        t = np.load(os.path.join(self.root_dir , 'vt.npy'))
        return x, t

    def reset(self):
        self.batch_index = 0

class EventLoader:
    def __init__(self, dataset, extractor, batch_size, mode='sample', fraction_positive=.5):
        """ 
        In between Base and Preproc.
        Loads extracted Events from disk, and does extraction
        note: a lot faster than Base because it doesn't need to to extraction
        """
        self.dataset = dataset # not used
        self.root_dir = os.path.join(os.path.join(data_dir, self.dataset.config['preproc_dir']), str(self.dataset.fold_index))
        self.extractor = extractor 
        self.batch_size = batch_size # not used
        self.fraction_positive = fraction_positive
        self.mode = mode
        self.batch_index = 0
        self.dataset_size = self.dataset.config['n_batches']

    def next(self):
        """
        """
        if self.mode == 'sample':
            return self.next_batch()
        elif self.mode == 'valid':
            return self.load_valid()
        else:
            raise Exception('unknown loader mode')



    def next_batch(self, extract=True):
        if self.batch_index==self.dataset_size:
            return None
        ## a bit hacky -- temporarily used before dataset fixed..probably can delete when you see this
        while True:
            try:
                x = np.load(os.path.join(self.root_dir , '%ix.npy'%self.batch_index))
                if extract:
                    x = self.extractor.extract_batch(x)
                t = np.load(os.path.join(self.root_dir , '%it.npy'%self.batch_index))
            except ExtractorException:
                self.batch_index += 1
            else:
                break
        self.batch_index += 1
        return x, t

    def load_valid(self, extract=True):
        x = np.load(os.path.join(self.root_dir , 'vx.npy'))
        if extract:
            x = self.extractor.extract_batch(x)
        t = np.load(os.path.join(self.root_dir , 'vt.npy'))
        return x, t

    def reset(self):
        self.batch_index = 0
if __name__ == '__main__':
    from sportvu.data.dataset import BaseDataset
    from sportvu.data.extractor import BaseExtractor
    dataset = BaseDataset('config/train_rev0.yaml', 0, load_raw=False)
    extractor = BaseExtractor('config/train_rev0.yaml')
    # loader = BaseLoader(dataset, extractor, 32, fraction_positive=0.5)
    # # batch = loader.next_batch(False)
    # # for ind, e in enumerate(batch):
    # #     print ind
    # #     try:
    # #         e.show('/u/wangkua1/Pictures/vis/%i.mp4' % ind)
    # #     except EventException:
    # #         pass
    # from tqdm import tqdm 
    # # for i in tqdm(range(10)):
    # #     b = loader.next()
    # v = loader.load_valid(True)
    # b = loader.next()
    

    loader = PreprocessedLoader(dataset, extractor, 32, fraction_positive=0.5)
    x, t = loader.next()    