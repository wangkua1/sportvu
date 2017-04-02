from __future__ import division
import cPickle as pickle
import yaml
import os
from sportvu import data
import numpy as np
game_dir = data.constant.game_dir


class BaseDataset:
    """base class for loading the dataset
    """

    def __init__(self, f_config, fold_index):
        # configuration
        self.config = yaml.load(open(f_config, 'rb'))
        assert (fold_index >= 0 and fold_index <
                self.config['data_config']['N_folds'])
        self.annotations = pickle.load(
            open(data.constant.data_dir + self.config['data_config']['annotation']))
        if self.config['data_config']['shuffle']:
            np.random.seed(self.config['randseed'])
            np.random.shuffle(self.annotations)
        N = len(self.annotations)
        val_start = np.round(
            fold_index / self.config['data_config']['N_folds'] * N).astype('int32')
        val_end = np.round((fold_index + 1) /
                           self.config['data_config']['N_folds'] * N).astype('int32')
        self.val_annotations = self.annotations[val_start:val_end]
        self.train_annotations = self.annotations[
            :val_start] + self.annotations[val_end:]

        self._make_annotation_dict()
        self.tfr = self.config['data_config']['tfr']
        self.t_jitter = self.config['data_config']['t_jitter']
        self.t_negative = self.config['data_config']['t_negative']
        self.game_ids = self.config['data_config']['game_ids']
        ###
        self.games = {}
        for gameid in self.game_ids:
            with open(os.path.join(game_dir, gameid + '.pkl'), 'rb') as f:
                raw_data = pickle.load(f)
            self.games[raw_data['gameid']] = raw_data

    def _make_annotation_dict(self):
        self.annotation_dict = {}
        for anno in self.annotations:
            if anno['gameid'] not in self.annotation_dict:
                self.annotation_dict[anno['gameid']] = {}
            if anno['quarter'] not in self.annotation_dict[anno['gameid']]:
                self.annotation_dict[anno['gameid']][anno['quarter']] = []
            self.annotation_dict[anno['gameid']][
                anno['quarter']].append(anno['gameclock'])
        for game in self.annotation_dict.values():
            for quarter_ind in game.keys():
                game[quarter_ind] = np.sort(game[quarter_ind])[::-1]

    def propose_positive_Ta(self, jitter=True, train=True):
        if train:
            while True:
                r_ind = np.random.randint(0, len(self.train_annotations))
                if not jitter:
                    ret =  self.train_annotations[r_ind]
                else:
                    anno = self.train_annotations[r_ind].copy()
                    anno['gameclock'] += np.random.rand() * self.t_jitter
                    ret = anno
                # check not too close to boundary (i.e. not enough frames to make a sequence)
                e = self.games[anno['gameid']]['events'][ret['eid']]
                for idx, moment in enumerate(e['moments']):
                    #seek
                    if moment[2] < ret['gameclock']:
                        if idx + self.tfr <= len(e['moments']) and idx - self.tfr >= 0:
                            return ret
                        else:
                            break # try again...

        else:
            raise Exception("not in training mode")

    def _make_annotation(self, gameid, quarter, gameclock,eid):
        return {'gameid': str(gameid), 'quarter': int(quarter), 
                'gameclock': float(gameclock), 'eid': int(eid)}

    def propose_Ta(self):
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

    def propose_negative_Ta(self):
        while True:
            cand = self.propose_Ta()
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


if __name__ == '__main__':
    dataset = BaseDataset('config/train_rev0.yaml', 0)
