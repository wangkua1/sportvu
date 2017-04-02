from __future__ import division
import cPickle as pickle
import yaml
import os
from sportvu import data
import numpy as np
game_dir = data.constant.game_dir


class ExtractorException(Exception):
    pass


def create_circle(radius):
    r_squared = np.arange(-radius, radius + 1)**2
    dist_to = r_squared[:, None] + r_squared
    # ones_circle = (dist_to <= radius**2).astype('float32')
    ones_circle = 1 - dist_to.astype('float32')**2 / ((radius + 1) * 2)**2
    circle_length = radius * 2 + 1
    return ones_circle, int(circle_length)


def pictorialize(xx, sample_rate=1, Y_RANGE=100, X_RANGE=50, radius=3):
    """
    xx of shape (..., 2)
    return: shape (..., Y_RANGE, X_RANGE) one hot encoded pictures
    WARNING: should really try to vectorize ***
    """
    # some preprocessing to make sure data is within range
    if Y_RANGE == X_RANGE:
        xx[xx >= Y_RANGE] = Y_RANGE - 1
    ###
    xx = np.array(xx).astype('int32')
    old_shape = list(xx.shape)
    Y_RANGE = Y_RANGE / sample_rate
    X_RANGE = X_RANGE / sample_rate
    # reasons behind padding radius is to avoid boundary cases when filling
    # with circles
    target = np.zeros(
        (old_shape[:-1] + [int(Y_RANGE + 2 * radius), int(X_RANGE + 2 * radius)]))
    nr_xx = xx.reshape(-1, xx.shape[-1])
    nr_target = target.reshape(-1, target.shape[-2], target.shape[-1])
    # create the small circle first
    ones_circle, circle_length = create_circle(radius)
    circles = ones_circle
    # fill it up
    ind0 = np.arange(nr_target.shape[0]).astype('int32')
    start_x = (nr_xx[:, 0] / sample_rate).astype('int32')
    start_y = (nr_xx[:, 1] / sample_rate).astype('int32')
    for ind in xrange(len(ind0)):  # WARNING ***
        nr_target[ind0[ind], start_x[ind]:start_x[ind] + circle_length,
                  start_y[ind]:start_y[ind] + circle_length] = circles
    if radius > 0:
        nr_target = nr_target[:, radius:-radius,
                              radius:-radius]  # shave off the padding
    target = nr_target.reshape((old_shape[:-1] + [int(Y_RANGE), int(X_RANGE)]))
    return target


def pictorialize_team(xx, sample_rate=1, Y_RANGE=100, X_RANGE=50, radius=0):
    """
    LEGACY FUNCTION
    please use caller_f_team to achieve team processing
    """
    """
    xx of shape (..., 2*N_PLAYERS)
    basically calls pictorialize N_PLAYERS times and combine the results
    """
    rolled_xx = np.rollaxis(xx, -1)
    for sli in xrange(int(rolled_xx.shape[0] / 2)):
        player = rolled_xx[2 * sli:2 * (sli + 1)]
        tmp = pictorialize(np.rollaxis(player, 0, len(
            player.shape)), sample_rate, Y_RANGE, X_RANGE, radius)
        retval = retval + tmp if sli > 0 else tmp
    return retval


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

    def __init__(self, d0flip=True, d1flip=True, jitter=[3, 3]):
        self.augment = True
        self.d0flip = d0flip
        self.d1flip = d1flip
        self.jitter = jitter

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
        return [ball, offense, defense]

    def extract(self, event):
        x = self.extract_raw(event)
        ctxy = []
        if self.augment and np.sum(self.jitter) > 0:
            d0_jit = (np.random.rand() * 2 - 1) * self.jitter[0]
            d1_jit = (np.random.rand() * 2 - 1) * self.jitter[1]
            jit = np.array([d0_jit, d1_jit])
            jit = jit.reshape(1, 2).repeat(len(x[0][0]), axis=0)
            for team in x:
                for player in team:
                    try:
                        player=np.array(player) + jit
                    except ValueError: # bad sequence where not all players have the same number of moments
                        raise ExtractorException()

        for play_sequence in x:
            try:
                team_matrix=np.concatenate(play_sequence, 1)
            except ValueError:
                raise ExtractorException()

            tm=pictorialize_team(team_matrix, sample_rate=1,
                                   Y_RANGE=100, X_RANGE=50,
                                   radius=0)
                
                
            ctxy.append(tm)
        ctxy = np.array(ctxy)
        ## compress the time dimension
        cxy = ctxy.sum(1)
        cxy[cxy>1] = 1

        if self.augment and self.d0flip and np.random.rand>.5:
            cxy = cxy[:,::-1]
        if self.augment and self.d1flip and np.random.rand>.5:
            cxy = cxy[:,:,::-1]
        return cxy

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
    ## concurrent
    import sys
    sys.path.append('/home/wangkua1/toolboxes/resnet')
    from resnet.utils.concurrent_batch_iter import ConcurrentBatchIterator
    from tqdm import tqdm
    
    ##
    dataset=BaseDataset('config/train_rev0.yaml', 0)
    extractor=BaseExtractor()
    loader=BaseLoader(dataset, extractor, 32, fraction_positive=1)
    print ('testing next_batch')
    batch=loader.next_batch()

    print ("compare loading latency")
    Q_size = 100
    N_thread = 32
    cloader = ConcurrentBatchIterator(loader, max_queue_size=Q_size, num_threads=N_thread)
    N = 100
    for i in tqdm(xrange(N), desc='multi thread Q size %i, N thread %i'%(Q_size, N_thread)):
        b = cloader.next()

    for i in tqdm(xrange(N), desc='single thread'):
        b = loader.next()

