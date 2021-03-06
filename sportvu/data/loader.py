from __future__ import division
import cPickle as pickle
import yaml
import os
from sportvu import data
from sportvu.vis.Event import Event, EventException
from sportvu.data.extractor import ExtractorException
import numpy as np
from utils import shuffle_2_array
from sportvu.utils import filter_discontinuous
game_dir = data.constant.game_dir
data_dir = data.constant.data_dir


class BaseLoader:
    def __init__(self, dataset, extractor, batch_size, mode='sample', fraction_positive=.5):
        self.dataset = dataset
        self.extractor = extractor
        self.batch_size = batch_size
        self.fraction_positive = fraction_positive
        self.mode = mode
        self.event_index = 0

    def next(self):
        """
        """
        if self.mode == 'sample':
            return self.next_batch()
        elif self.mode == 'valid':
            return self.load_valid()
        else:
            raise Exception('unknown loader mode')

    def next_batch(self, extract=True, no_anno=False):
        N_pos = int(self.fraction_positive * self.batch_size)
        N_neg = self.batch_size - N_pos
        ret_val = []
        if not no_anno:
            func = [self.dataset.propose_positive_Ta,
                    self.dataset.propose_negative_Ta]
        else:
            func = [self.dataset.propose_Ta]
        Ns = [N_pos, N_neg]
        # anno = func[0]()
        for j in xrange(len(func)):
            for _ in xrange(Ns[j]):
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
                            # just to make sure event not malformed (like
                            # missing player)
                            _ = self.extractor.extract_raw(e)
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

    def load_by_annotations(self, annotations, extract=True):
        """
        no labels returned
        """
        ret_val = []
        ret_labels = []
        self.extractor.augment = False
        for anno in annotations:
            try:
                e = Event(self.dataset.games[anno['gameid']][
                          'events'][anno['eid']], gameid=anno['gameid'])
                e.sequence_around_t(
                    anno['gameclock'], self.dataset.tfr)  # EventException
                if extract:
                    # ExtractorException
                    ret_val.append(self.extractor.extract(e))
                else:
                    # just to make sure event not malformed (like
                    # missing player)
                    _ = self.extractor.extract_raw(e)
                    ret_val.append(e)
            except EventException as exc:
                continue
            except ExtractorException as exc:
                continue
        return ret_val

    def load_split(self, split='val', extract=True, positive_only=False):
        N_pos = 0
        ret_val = []
        ret_labels = []
        self.extractor.augment = False
        istrain = split == 'train'
        while True:
            anno = self.dataset.propose_positive_Ta(
                jitter=False, train=istrain, loop=True)
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
                    # just to make sure event not malformed (like
                    # missing player)
                    _ = self.extractor.extract_raw(e)
                    ret_val.append(e)
            except EventException as exc:
                continue
            except ExtractorException as exc:
                continue
            else:
                N_pos += 1
                ret_labels.append([0, 1])
        if not positive_only:
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
                            # just to make sure event not malformed (like
                            # missing player)
                            _ = self.extractor.extract_raw(e)
                            ret_val.append(e)
                    except EventException as exc:
                        pass
                    except ExtractorException as exc:
                        pass
                    else:
                        ret_labels.append([1, 0])
                        break
        self.extractor.augment = True
        return (np.array(ret_val),
                np.array(ret_labels)
                )

    def load_train(self, extract, positive_only):
        return self.load_split(split='train', extract=extract, positive_only=positive_only)

    def load_valid(self, extract=True, positive_only=False):
        return self.load_split(split='val', extract=extract, positive_only=positive_only)

    def _load_event(self, event_id, game_id, extract, every_K_frame, player_id=None, dont_resolve_basket=False):
        ret_val = []
        ret_gameclocks = []
        ret_frame_idx = []
        e = Event(self.dataset.games[game_id]['events'][
            event_id], gameid=game_id)
        N_moments = len(e.moments)
        for i in xrange(0, N_moments, every_K_frame):
            try:
                e = Event(self.dataset.games[game_id]['events'][
                          event_id], gameid=game_id)
                game_clock = e.moments[i].game_clock
                e.sequence_around_t(
                    game_clock, self.dataset.tfr)  # EventException
                # just to make sure event not malformed (like
                # missing player)
                _ = self.extractor.extract_raw(e)
                ret_val.append(e)
                ret_gameclocks.append(game_clock)
                ret_frame_idx.append(i)
            except EventException as exc:
                continue
            except ExtractorException as exc:
                continue
        if len(ret_val) == 0:  # malformed Event
            ret = 0
        else:
            if extract:
                ret_val = self.extractor.extract_batch(ret_val, player_id=player_id,dont_resolve_basket=dont_resolve_basket)
            ret = [ret_val, ret_gameclocks, ret_frame_idx]
        return ret

    def load_split_event(self, split, extract, every_K_frame=4):
        if split == 'val':
            split_hash = self.dataset.val_hash
        elif split == 'train':
            split_hash = self.dataset.train_hash
        else:
            raise NotImplementedError()
        if self.event_index == len(split_hash):
            self.event_index = 0
            return None
        vh = split_hash.values()[self.event_index]

        ret_labels = filter(lambda t: t != -1, [i['gameclock'] for i in vh])
        self.extractor.augment = False
        event_id = vh[0]['eid']
        game_id = vh[0]['gameid']
        ret = self._load_event(event_id, game_id, extract, every_K_frame)
        if ret == 0:  # malformed Event
            ret = 0
        else:
            ret_val, ret_gameclocks, ret_frame_idx = ret
            meta = [vh[0]['eid'], vh[0]['gameid']]
            ret = [ret_val, ret_labels, ret_gameclocks, meta]
        self.extractor.augment = True
        self.event_index += 1
        return ret

    def load_event(self, game_id, event_id, every_K_frame, player_id=None):
        o = self.extractor.augment
        self.extractor.augment = False
        ret = self._load_event(event_id, game_id, True, every_K_frame, player_id=player_id, dont_resolve_basket=True)
        self.extractor.augment = o
        return ret



    def reset(self):
        pass


class PreprocessedLoader:
    def __init__(self, dataset, extractor, batch_size, mode='sample', fraction_positive=.5):
        """ 
        simply loads numpy matrices from disk without preprocessing
        """
        self.dataset = dataset  # not used
        self.root_dir = os.path.join(os.path.join(data_dir, self.dataset.config[
                                     'preproc_dir']), str(self.dataset.fold_index))
        self.extractor = extractor  # not used
        self.batch_size = batch_size  # not used
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
        if self.batch_index == self.dataset_size:
            return None
        x = np.load(os.path.join(self.root_dir, '%ix.npy' % self.batch_index))
        t = np.load(os.path.join(self.root_dir, '%it.npy' % self.batch_index))
        self.batch_index += 1
        return x, t

    def load_valid(self, extract=True):
        x = np.load(os.path.join(self.root_dir, 'vx.npy'))
        t = np.load(os.path.join(self.root_dir, 'vt.npy'))
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
        self.dataset = dataset  # not used
        self.root_dir = os.path.join(os.path.join(data_dir, self.dataset.config[
                                     'preproc_dir']), str(self.dataset.fold_index))
        self.extractor = extractor
        self.batch_size = batch_size  # not used
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
        if self.batch_index == self.dataset_size:
            return None
        # a bit hacky -- temporarily used before dataset fixed..probably can
        # delete when you see this
        while True:
            try:
                x = np.load(os.path.join(self.root_dir, '%ix.npy' %
                                         self.batch_index))
                if extract:
                    x = self.extractor.extract_batch(x)
                t = np.load(os.path.join(self.root_dir, '%it.npy' %
                                         self.batch_index))
            except ExtractorException:
                self.batch_index += 1
            else:
                break
        self.batch_index += 1
        return x, t

    def load_valid(self, extract=True):
        x = np.load(os.path.join(self.root_dir, 'vx.npy'))
        if extract:
            x = self.extractor.extract_batch(x)
        t = np.load(os.path.join(self.root_dir, 'vt.npy'))
        return x, t

    def reset(self):
        self.batch_index = 0


class SequenceLoader:
    def __init__(self, dataset, extractor, batch_size, mode='sample',
                 fraction_positive=.5, negative_fraction_hard=0):
        """ 
        """
        self.negative_fraction_hard = negative_fraction_hard
        self.dataset = dataset  # not used
        self.root_dir = os.path.join(os.path.join(data_dir, self.dataset.config[
                                     'preproc_dir']), str(self.dataset.fold_index))
        self.extractor = extractor
        self.batch_size = batch_size  # not used
        self.fraction_positive = fraction_positive
        self.mode = mode
        self.batch_index = 0
        # self.dataset_size = self.dataset.config['n_batches']
        self.pos_x = np.load(os.path.join(self.root_dir, 'pos_x.npy'))
        self.neg_x = np.load(os.path.join(self.root_dir, 'neg_x.npy'))
        self.pos_t = np.load(os.path.join(self.root_dir, 'pos_t.npy'))
        # self.neg_t = np.load(os.path.join(self.root_dir, 'neg_t.npy'))
        self.neg_t = np.array([[1, 0]]).repeat(self.neg_x.shape[0], axis=0)
        if self.negative_fraction_hard > 0:
            self.hard_neg_x = np.load(os.path.join(
                self.root_dir, 'hard_neg_x.npy'))
            self.hard_neg_t = np.load(os.path.join(
                self.root_dir, 'hard_neg_t.npy'))
        self.val_x = np.load(os.path.join(self.root_dir, 'vx.npy'))
        self.val_t = np.load(os.path.join(self.root_dir, 'vt.npy'))
        self.pos_ind = 0
        self.neg_ind = 0
        self.N_pos = int(batch_size * fraction_positive)
        self.N_neg = batch_size - self.N_pos
        if self.negative_fraction_hard > 0:
            self.N_hard_neg = int(self.N_neg * negative_fraction_hard)
            self.N_neg = self.N_neg - self.N_hard_neg
            self.hard_neg_ind = 0

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
        # if self.batch_index == self.dataset_size:
        #     return None
        if self.pos_ind + self.N_pos >= self.pos_x.shape[0]:
            self.pos_ind = 0
            self.pos_x, self.pos_t = shuffle_2_array(self.pos_x, self.pos_t)
        if self.neg_ind + self.N_neg >= self.neg_x.shape[0]:
            self.neg_ind = 0
            self.neg_x, self.neg_t = shuffle_2_array(self.neg_x, self.neg_t)
        if (self.negative_fraction_hard > 0 and
                self.hard_neg_ind + self.N_hard_neg >= self.hard_neg_x.shape[0]):
            self.hard_neg_ind = 0
            self.hard_neg_x, self.hard_neg_t = shuffle_2_array(
                self.hard_neg_x, self.hard_neg_t)

        s = list(self.pos_x.shape)
        s[0] = self.batch_size
        x = np.zeros(s)
        t = np.zeros((self.batch_size, 2))
        x[:self.N_pos] = self.pos_x[self.pos_ind:self.pos_ind + self.N_pos]
        t[:self.N_pos] = self.pos_t[self.pos_ind:self.pos_ind + self.N_pos]
        if not self.negative_fraction_hard > 0:
            x[self.N_pos:] = self.neg_x[self.neg_ind:self.neg_ind + self.N_neg]
            t[self.N_pos:] = self.neg_t[self.neg_ind:self.neg_ind + self.N_neg]
        else:
            x[self.N_pos:self.N_pos + self.N_hard_neg] = self.hard_neg_x[
                self.hard_neg_ind:self.hard_neg_ind + self.N_hard_neg]
            t[self.N_pos:self.N_pos + self.N_hard_neg] = self.hard_neg_t[
                self.hard_neg_ind:self.hard_neg_ind + self.N_hard_neg]
            x[self.N_pos + self.N_hard_neg:] = self.neg_x[self.neg_ind:self.neg_ind + self.N_neg]
            t[self.N_pos + self.N_hard_neg:] = self.neg_t[self.neg_ind:self.neg_ind + self.N_neg]
        if extract:
            x = self.extractor.extract_batch(x, input_is_sequence=True)
        self.pos_ind += self.N_pos
        self.neg_ind += self.N_neg
        if self.negative_fraction_hard > 0:
            self.hard_neg_ind += self.N_hard_neg
        return x, t

    def load_valid(self, extract=True):
        x = self.val_x
        if extract:
            x = self.extractor.extract_batch(x, input_is_sequence=True)
        t = self.val_t
        return x, t

    def reset(self):
        self.batch_index = 0


class Seq2SeqLoader:
    def __init__(self, dataset, extractor, batch_size, mode='sample',
                 fraction_positive=.5, negative_fraction_hard=0,
                 use_filter_discontinuous=True, move_N_neg_to_val=0):
        """ 
        """
        self.negative_fraction_hard = negative_fraction_hard
        self.dataset = dataset  # not used
        self.root_dir = os.path.join(os.path.join(data_dir, self.dataset.config[
                                     'preproc_dir']), str(self.dataset.fold_index))
        self.extractor = extractor
        self.batch_size = batch_size  # not used
        self.fraction_positive = fraction_positive
        self.mode = mode
        self.batch_index = 0
        # self.dataset_size = self.dataset.config['n_batches']
        self.pos_x = np.load(os.path.join(self.root_dir, 'pos_x.npy'))
        self.neg_x = np.load(os.path.join(self.root_dir, 'neg_x.npy'))
        self.pos_t = np.load(os.path.join(self.root_dir, 'pos_t.npy'))
        # self.neg_t = np.load(os.path.join(self.root_dir, 'neg_t.npy'))
        self.neg_t = np.array([[1, 0]]).repeat(self.neg_x.shape[0], axis=0)
        if self.negative_fraction_hard > 0:
            self.hard_neg_x = np.load(os.path.join(
                self.root_dir, 'hard_neg_x.npy'))
            self.hard_neg_t = np.load(os.path.join(
                self.root_dir, 'hard_neg_t.npy'))
        self.val_x = np.load(os.path.join(self.root_dir, 'vx.npy'))
        self.val_t = np.load(os.path.join(self.root_dir, 'vt.npy'))
        self.pos_ind = 0
        self.neg_ind = 0
        self.N_pos = int(batch_size * fraction_positive)
        self.N_neg = batch_size - self.N_pos
        if self.negative_fraction_hard > 0:
            self.N_hard_neg = int(self.N_neg * negative_fraction_hard)
            self.N_neg = self.N_neg - self.N_hard_neg
            self.hard_neg_ind = 0
        self.val_ind = 0
        if use_filter_discontinuous:
            self.pos_x = filter_discontinuous(self.pos_x)
            self.val_x = filter_discontinuous(self.val_x)
            self.neg_x = filter_discontinuous(self.neg_x)
            if self.negative_fraction_hard > 0:
                self.hard_neg_x = filter_discontinuous(self.hard_neg_x)
        if move_N_neg_to_val > 0:
            self.val_x = np.concatenate(
                [self.val_x, self.neg_x[:move_N_neg_to_val]], axis=0)
            self.neg_x = self.neg_x[move_N_neg_to_val:]

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
        # if self.batch_index == self.dataset_size:
        #     return None
        if self.pos_ind + self.N_pos >= self.pos_x.shape[0]:
            self.pos_ind = 0
            self.pos_x, self.pos_t = shuffle_2_array(self.pos_x, self.pos_t)
        if self.neg_ind + self.N_neg >= self.neg_x.shape[0]:
            self.neg_ind = 0
            self.neg_x, self.neg_t = shuffle_2_array(self.neg_x, self.neg_t)
        if (self.negative_fraction_hard > 0 and
                self.hard_neg_ind + self.N_hard_neg >= self.hard_neg_x.shape[0]):
            self.hard_neg_ind = 0
            self.hard_neg_x, self.hard_neg_t = shuffle_2_array(
                self.hard_neg_x, self.hard_neg_t)

        s = list(self.pos_x.shape)
        s[0] = self.batch_size
        x = np.zeros(s)
        x[:self.N_pos] = self.pos_x[self.pos_ind:self.pos_ind + self.N_pos]
        if not self.negative_fraction_hard > 0:
            x[self.N_pos:] = self.neg_x[self.neg_ind:self.neg_ind + self.N_neg]
        else:
            x[self.N_pos:self.N_pos + self.N_hard_neg] = self.hard_neg_x[
                self.hard_neg_ind:self.hard_neg_ind + self.N_hard_neg]
            x[self.N_pos + self.N_hard_neg:] = self.neg_x[self.neg_ind:self.neg_ind + self.N_neg]
        ###

        if extract:
            ret = self.extractor.extract_batch(x, input_is_sequence=True)
        self.pos_ind += self.N_pos
        self.neg_ind += self.N_neg
        if self.negative_fraction_hard > 0:
            self.hard_neg_ind += self.N_hard_neg
        return ret

    def load_valid(self, extract=True):
        if self.val_ind + self.batch_size > len(self.val_x):
            self.val_ind = 0
            return None
        x = self.val_x[self.val_ind:self.val_ind + self.batch_size]
        if extract:
            ret = self.extractor.extract_batch(x, input_is_sequence=True)
        self.val_ind += self.batch_size
        return ret

    def reset(self):
        self.batch_index = 0


if __name__ == '__main__':
    # from sportvu.data.dataset import BaseDataset
    # from sportvu.data.extractor import BaseExtractor
    # dataset = BaseDataset('config/train_rev0.yaml', 0, load_raw=False)
    # extractor = BaseExtractor('config/train_rev0.yaml')
    # # loader = BaseLoader(dataset, extractor, 32, fraction_positive=0.5)
    # # # batch = loader.next_batch(False)
    # # # for ind, e in enumerate(batch):
    # # #     print ind
    # # #     try:
    # # #         e.show('/u/wangkua1/Pictures/vis/%i.mp4' % ind)
    # # #     except EventException:
    # # #         pass
    # # from tqdm import tqdm
    # # # for i in tqdm(range(10)):
    # # #     b = loader.next()
    # # v = loader.load_valid(True)
    # # b = loader.next()

    # loader = PreprocessedLoader(dataset, extractor, 32, fraction_positive=0.5)
    # x, t = loader.next()

    from sportvu.data.dataset import BaseDataset
    from sportvu.data.extractor import BaseExtractor
    dataset = BaseDataset('config/rev2.yaml', 0, load_raw=False)
    extractor = BaseExtractor('config/rev2.yaml')
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

    loader = SequenceLoader(dataset, extractor, 32, fraction_positive=0.5)
    x, t = loader.next()
