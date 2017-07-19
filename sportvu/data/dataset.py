from __future__ import division
import cPickle as pickle
import yaml
import os
from sportvu import data
import numpy as np
game_dir = data.constant.game_dir
from sportvu.vis.Event import Event, EventException


def _hash(i):
    return int(i['gameid']) + 1000 * i['eid']


def disentangle_train_val(train, val):
    """
    Given annotations of train/val splits, make sure no overlapping Event
    -> for later detection testing
    """
    # simple hash := gameid + 1000*eid
    assert(np.max([i['eid'] for i in train] + [i['eid'] for i in val]) < 1000)
    new_train = []
    new_val = []
    while len(val) > 0:
        ve = val.pop()
        vh = _hash(ve)
        if vh in [_hash(i) for i in train] + [_hash(i) for i in new_train]:
            new_train.append(ve)
            # to balance, find a unique train_anno to put in val
            while True:
                te = train.pop(0)
                if _hash(te) in [_hash(i) for i in train]:  # not unique, put back
                    train.append(te)
                else:
                    new_val.append(te)
                    break
        else:
            new_val.append(ve)
    new_train += train
    return new_train, new_val


class BaseDataset:
    """base class for loading the dataset
    """

    def __init__(self, f_config, fold_index, load_raw=True, no_anno=False):
        # configuration
        self.fold_index = fold_index
        if type(f_config) ==str:
            self.config = yaml.load(open(f_config, 'rb'))
        else:
            self.config = f_config
        assert (fold_index >= 0 and fold_index <
                self.config['data_config']['N_folds'])
        self.tfr = self.config['data_config']['tfr']
        self.t_jitter = self.config['data_config']['t_jitter']
        self.t_negative = self.config['data_config']['t_negative']
        self.game_ids = self.config['data_config']['game_ids']
        ###
        if load_raw == True:
            self.games = {}
            for gameid in self.game_ids:
                with open(os.path.join(game_dir, gameid + '.pkl'), 'rb') as f:
                    raw_data = pickle.load(f)
                self.games[raw_data['gameid']] = raw_data
        if not no_anno:
            self.annotations = pickle.load(
                open(data.constant.data_dir + self.config['data_config']['annotation']))
            # Hacky (human labellers has some delay, so usually they label a bit after a pnr)
            # here we adjust for it...the way I selected this was just by
            # visualizing labels
            for anno in self.annotations:
                anno['gameclock'] += .6  # + means earlier in gameclock
            self.train_annotations, self.val_annotations = self._split(
                                        self.annotations, fold_index)
            # make sure no overlapping Event between train and val
            self.train_annotations, self.val_annotations = disentangle_train_val(
                self.train_annotations, self.val_annotations)
            self._make_annotation_dict()
            # hard negative examples
            ## WARNING: K-fold not supported here
            ##  user needs to make sure the following pkl
            ##  comes from the correct fold
            if 'hard-negatives' in self.config['data_config']:
                self.hard_negatives = pickle.load(
                    open(data.constant.data_dir + self.config['data_config']['hard-negatives']))
            if load_raw:
                # annotation only has Events with PNR
                # need to find Events without PNR, split into train/val
                #   for negatives examples
                # Let's call these Events 'void'
                # use gameclock = -1
                self.voids = []  
                for gameid in self.game_ids:
                    curr_game_events = self.games[gameid]['events']
                    for eid, event in enumerate(curr_game_events):
                        if not eid in self.annotation_dict_eids[gameid][event['quarter']]:
                            self.voids.append(self._make_annotation(
                                gameid, event['quarter'], -1, eid))
                self.train_voids, self.val_voids = self._split(
                                            self.voids, fold_index)

                # loader use below for detection task
                # hash validation events
                self.val_hash = {}
                for va in self.val_annotations + self.val_voids:
                    k = _hash(va)
                    if k not in self.val_hash:
                        self.val_hash[k] = []
                    self.val_hash[k].append(va)
                # hash train events for mining hard examples
                self.train_hash = {}
                for va in self.train_annotations + self.train_voids:
                    k = _hash(va)
                    if k not in self.train_hash:
                        self.train_hash[k] = []
                    self.train_hash[k].append(va)
            
        self.val_ind = 0
        self.train_ind = 0

    def _split(self, annotations, fold_index):
        if self.config['data_config']['shuffle']:
            np.random.seed(self.config['randseed'])
            np.random.shuffle(annotations)
        N = len(annotations)
        val_start = np.round(
            fold_index / self.config['data_config']['N_folds'] * N).astype('int32')
        val_end = np.round((fold_index + 1) /
                           self.config['data_config']['N_folds'] * N).astype('int32')
        val_annotations = annotations[val_start:val_end]
        train_annotations = annotations[
            :val_start] + annotations[val_end:]
        return train_annotations, val_annotations

    def _make_annotation_dict(self):
        self.annotation_dict = {}
        self.annotation_dict_eids = {}
        for anno in self.annotations:
            if anno['gameid'] not in self.annotation_dict:
                self.annotation_dict[anno['gameid']] = {}
                self.annotation_dict_eids[anno['gameid']] = {}
            if anno['quarter'] not in self.annotation_dict[anno['gameid']]:
                self.annotation_dict[anno['gameid']][anno['quarter']] = []
                self.annotation_dict_eids[anno['gameid']][anno['quarter']] = []
            self.annotation_dict[anno['gameid']][
                anno['quarter']].append(anno['gameclock'])
            self.annotation_dict_eids[anno['gameid']][
                anno['quarter']].append(anno['eid'])
        for game in self.annotation_dict.values():
            for quarter_ind in game.keys():
                game[quarter_ind] = np.sort(game[quarter_ind])[::-1]

    def propose_positive_Ta(self, jitter=True, train=True, loop=False):
        if not loop:  # sampling from training annotations
            while True:
                r_ind = np.random.randint(0, len(self.train_annotations))
                if not jitter:
                    ret = self.train_annotations[r_ind]
                else:
                    anno = self.train_annotations[r_ind].copy()
                    anno['gameclock'] += np.random.rand() * self.t_jitter
                    ret = anno

                # check not too close to boundary (i.e. not enough frames to
                # make a sequence)
                e = self.games[anno['gameid']]['events'][ret['eid']]
                for idx, moment in enumerate(e['moments']):
                    # seek
                    if moment[2] < ret['gameclock']:
                        if idx + self.tfr <= len(e['moments']) and idx - self.tfr >= 0:
                            return ret
                        else:
                            break  # try again...

        else:
            if train:
                if self.train_ind == len(self.train_annotations):  # end, reset
                    self.train_ind = 0
                    return None
                else:
                    ret = self.train_annotations[self.train_ind]
                    self.train_ind += 1
                    return ret
            else:
                if self.val_ind == len(self.val_annotations):  # end, reset
                    self.val_ind = 0
                    return None
                else:
                    ret = self.val_annotations[self.val_ind]
                    self.val_ind += 1
                    return ret

    def _make_annotation(self, gameid, quarter, gameclock, eid):
        return {'gameid': str(gameid), 'quarter': int(quarter),
                'gameclock': float(gameclock), 'eid': int(eid)}

    def propose_Ta(self, train=True):
        while True:
            g_ind = np.random.randint(0, len(self.games))
            e_ind = np.random.randint(
                0, len(self.games[self.games.keys()[g_ind]]['events']))

            e = self.games[self.games.keys()[g_ind]]['events'][e_ind]
            if len(e['moments']) < self.tfr * 2:
                continue
            m_ind = np.random.randint(self.tfr, len(e['moments']) - self.tfr)
            return self._make_annotation(self.games.keys()[g_ind],
                                         e['quarter'],
                                         e['moments'][m_ind][2],
                                         e_ind)

    def propose_negative_Ta(self, train=True):
        ## make sure it's in the right split
        if train:
            annos = self.train_annotations + self.train_voids
        else:
            annos = self.val_annotations + self.val_voids
        split_hash = [_hash(anno) for anno in annos]
        while True:
            cand = self.propose_Ta()
            if _hash(cand) not in split_hash:
                continue
            g_cand = cand['gameclock']
            positives = self.annotation_dict[cand['gameid']][cand['quarter']]
            # too close to a positive label
            if np.min(np.abs(positives - g_cand)) < self.t_negative:
                # print ('proporal positive')
                continue
            return cand

    # def prepare_train_batch(N, fraction_positive=.5):
    #     N_pos = np.round(N * fraction_positive)
    #     N_neg = N - N_pos
    #     for t_ind in xrange(N_pos):
    #         pass


class SeqDataset:
    def __init__(self, f_config, load_raw=True):
        # configuration
        # self.fold_index = fold_index
        if type(f_config) ==str:
            self.config = yaml.load(open(f_config, 'rb'))
        else:
            self.config = f_config
        self.min_moments = self.config['data_config']['min_moments']
        self.train_game_ids = self.config['data_config']['train_game_ids']
        self.val_game_ids = self.config['data_config']['val_game_ids']
        self.game_ids = self.train_game_ids+self.val_game_ids
        if load_raw:
            self.games = {}
            for gameid in self.game_ids:
                with open(os.path.join(game_dir, gameid + '.pkl'), 'rb') as f:
                    raw_data = pickle.load(f)
                self.games[raw_data['gameid']] = raw_data

    def propose_Ta(self, train=True, return_Event=False):
        if train:
            game_ids = self.train_game_ids
        else:
            game_ids = self.val_game_ids
        while True:
            g_ind = np.random.randint(0, len(game_ids))
            e_ind = np.random.randint(
                0, len(self.games[game_ids[g_ind]]['events']))

            e = self.games[game_ids[g_ind]]['events'][e_ind]
            if len(e['moments']) < self.min_moments:
                continue
            if return_Event:
                return Event(e,gameid=game_ids[g_ind])
            else:
                return e

    def get_next_val(self):
        for game_id in self.val_game_ids:
            for e in self.games[game_id]['events']:
                yield e

if __name__ == '__main__':
    f_data_config = '/home/ethanf/projects/sportvu/sportvu/data/config/ethanf_seq.yaml'
    dataset = SeqDataset(f_data_config, 0)
