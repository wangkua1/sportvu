from __future__ import division
import cPickle as pickle
import yaml
import os
from sportvu import data
import numpy as np
import yaml
from utils import pictorialize_team, pictorialize_fast, make_3teams_11players
game_dir = data.constant.game_dir


class ExtractorException(Exception):
    pass


class BaseExtractor:
    """base class for sequence extraction
        Input: a truncated Event
        Output: classifier model input

    Simplest possible parametrization, a collapsed image of the full court
    Options (all 3 can be used in conjunction):
        -d0flip
        -d1flip
        -jitter (x,y)
    returns a 3 channel image of (ball, offense, defense)
            -> resolves possession: which side of the court it's one
    """

    def __init__(self, f_config):
        self.augment = True
        self.config = yaml.load(open(f_config, 'rb'))['extractor_config']

    def extract_raw(self, event):
        """
        """
        ##
        moments = event.moments
        off_is_home = event.is_home_possession(moments[len(moments) // 2])
        ball, offense, defense = [[]], [
            [], [], [], [], []], [[], [], [], [], []]
        for moment in moments:
            ball[0].append([moment.ball.x, moment.ball.y])
            off_id, def_id = 0, 0
            for player in moment.players:
                if (player.team.id == event.home_team_id) == off_is_home:  # offense
                    offense[off_id].append([player.x, player.y])
                    off_id += 1
                else:  # defense
                    defense[def_id].append([player.x, player.y])
                    def_id += 1
        if not ((len(np.array(ball).shape) == 3
                 and len(np.array(offense).shape) == 3
                 and len(np.array(defense).shape) == 3)
                and
                (np.array(ball).shape[1] == np.array(offense).shape[1]
                 and np.array(offense).shape[1] == np.array(defense).shape[1])):
            raise ExtractorException()
        return [ball, offense, defense]

    def extract(self, event):
        x = self.extract_raw(event)
        ctxy = []
        if self.augment and np.sum(self.config['jitter']) > 0:
            d0_jit = (np.random.rand() * 2 - 1) * self.config['jitter'][0]
            d1_jit = (np.random.rand() * 2 - 1) * self.config['jitter'][1]
            jit = np.array([d0_jit, d1_jit])
            jit = jit.reshape(1, 2).repeat(len(x[0][0]), axis=0)
            for team in x:
                for player in team:
                    try:
                        player = np.array(player) + jit
                    except ValueError:  # bad sequence where not all players have the same number of moments
                        raise ExtractorException()

        for play_sequence in x:
            try:
                team_matrix = np.concatenate(play_sequence, 1)
            except ValueError:
                raise ExtractorException()

            tm = pictorialize_team(team_matrix, sample_rate=self.config['sample_rate'],
                                   Y_RANGE=self.config[
                                       'Y_RANGE'], X_RANGE=self.config['X_RANGE'],
                                   radius=self.config['radius'])

            ctxy.append(tm)
        ctxy = np.array(ctxy)
        if len(ctxy.shape) == 1:  # different teams have different length
            raise ExtractorException()
        # compress the time dimension
        if 'video' in self.config and self.config['video']:
            if self.augment and self.config['d0flip'] and np.random.rand > .5:
                ctxy = ctxy[:, :, ::-1]
            if self.augment and self.config['d1flip'] and np.random.rand > .5:
                ctxy = ctxy[:, :, :, ::-1]
            return ctxy
        else:
            cxy = ctxy.sum(1)
            cxy[cxy > 1] = 1
            if self.augment and self.config['d0flip'] and np.random.rand > .5:
                cxy = cxy[:, ::-1]
            if self.augment and self.config['d1flip'] and np.random.rand > .5:
                cxy = cxy[:, :, ::-1]
            return cxy

    def extract_batch(self, events_arr):
        sequences = np.array([make_3teams_11players(
            self.extract_raw(e)) for e in events_arr])
        if self.augment and np.sum(self.config['jitter']) > 0:
            d0_jit = (np.random.rand() * 2 - 1) * self.config['jitter'][0]
            d1_jit = (np.random.rand() * 2 - 1) * self.config['jitter'][1]
            # hacky: can delete after -- temporary for malformed data (i.e.
            # missing player)
            try:
                sequences[:, :, :, 0] += d0_jit
            except:
                raise ExtractorException()
            sequences[:, :, :, 1] += d1_jit
        ##
        bctxy = pictorialize_fast(sequences)
        # compress the time dimension
        if 'video' in self.config and self.config['video']:
            if self.augment and self.config['d0flip'] and np.random.rand > .5:
                bctxy = bctxy[:, :, :, ::-1]
            if self.augment and self.config['d1flip'] and np.random.rand > .5:
                bctxy = bctxy[:, :, :, :, ::-1]
            return bctxy
        else:
            bcxy = bctxy.sum(2)
            bcxy[bcxy > 1] = 1
            if self.augment and self.config['d0flip'] and np.random.rand > .5:
                bcxy = bcxy[:, :, ::-1]
            if self.augment and self.config['d1flip'] and np.random.rand > .5:
                bcxy = bcxy[:, :, :, ::-1]
            return bcxy


"""
HalfCourt Extractor
This extractor takes the Event
        1. flips basket so everything lives in a halfcourt
            (if sequence crossed both halves, it's not what we care about anyways
            , so it's okay to randomly chop them off)
"""
"""
Ball Extractor


"""
"""
ImagePyramid Extractor


"""
if __name__ == '__main__':
    from sportvu.data.dataset import BaseDataset
    from sportvu.data.extractor import BaseExtractor
    from loader import BaseLoader
    ##
    f_config = 'config/train_rev0.yaml'
    dataset = BaseDataset(f_config, 0)
    extractor = BaseExtractor(f_config)
    loader = BaseLoader(dataset, extractor, 35, fraction_positive=0)
    print ('testing next_batch')
    batch = loader.next_batch(extract=False)
    for eind, event in enumerate(batch[0]):
        event.show('/home/wangkua1/Pictures/vis/%i.mp4' % eind)

    # visualize model input
    # import  matplotlib.pyplot as plt
    # plt.ion()
    # for x in batch[0]:
    #     img = np.rollaxis(x, 0, 3)
    #     plt.imshow(img)
    #     raw_input()

    # ## concurrent
    # import sys
    # sys.path.append('/home/wangkua1/toolboxes/resnet')
    # from resnet.utils.concurrent_batch_iter import ConcurrentBatchIterator
    # from tqdm import tqdm

    # print ("compare loading latency")
    # Q_size = 100
    # N_thread = 32
    # cloader = ConcurrentBatchIterator(loader, max_queue_size=Q_size, num_threads=N_thread)
    # N = 100
    # for i in tqdm(xrange(N), desc='multi thread Q size %i, N thread %i'%(Q_size, N_thread)):
    #     b = cloader.next()

    # for i in tqdm(xrange(N), desc='single thread'):
    #     b = loader.next()
