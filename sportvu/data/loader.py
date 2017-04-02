from __future__ import division
import cPickle as pickle
import yaml
import os
from sportvu import data
from sportvu.vis.Event import Event, EventException
from sportvu.data.extractor import ExtractorException
import numpy as np
game_dir = data.constant.game_dir


class BaseLoader:
    def __init__(self, dataset, extractor, batch_size, mode='sample', fraction_positive=.5):
        self.dataset = dataset
        self.extractor = extractor
        self.batch_size = 32
        self.fraction_positive = fraction_positive
        self.mode = mode

    def next(self):
        """
        """
        if self.mode == 'sample':
            return self.next_batch()
        elif self.mode == 'pass':
            return self.next_pass()
        else:
            raise Exception('unknown loader mode')

    def next_batch(self, extract=True):
        N_pos = int(self.fraction_positive * self.batch_size)
        N_neg = self.batch_size - N_pos
        ret_val = []
        func = [self.dataset.propose_positive_Ta,
                self.dataset.propose_negative_Ta]
        Ns = [N_pos, N_neg]
        for j in xrange(len(Ns)):
            for i in xrange(Ns[j]):
                while True:
                    try:
                        anno = func[j]()
                        e = Event(self.dataset.games[anno['gameid']][
                                  'events'][anno['eid']], gameid=anno['gameid'])
                        e.sequence_around_t(
                            anno['gameclock'], self.dataset.tfr) # EventException
                        if extract:
                            ret_val.append(self.extractor.extract(e)) # ExtractorException
                        else:
                            ret_val.append(e)
                    except (EventException, ExtractorException) as exc:
                        pass
                    else:
                        break

                    
        return ret_val

    def next_pass(self):
        pass

    def reset(self):
        pass

if __name__ == '__main__':
    from sportvu.data.dataset import BaseDataset
    from sportvu.data.extractor import BaseExtractor
    dataset = BaseDataset('config/train_rev0.yaml', 0)
    extractor = BaseExtractor()
    loader = BaseLoader(dataset, extractor, 32, fraction_positive=0.5)
    batch = loader.next_batch(False)
    for ind, e in enumerate(batch):
        print ind
        try:
            e.show('/u/wangkua1/Pictures/vis/%i.mp4'%ind)
        except EventException:
            pass
        