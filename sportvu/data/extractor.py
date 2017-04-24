from __future__ import division
import cPickle as pickle
import yaml
import os
from sportvu import data
import numpy as np
import yaml
from utils import (pictorialize_team, pictorialize_fast, 
                make_3teams_11players, make_reference, scale_last_dim)
game_dir = data.constant.game_dir


class ExtractorException(Exception):
    pass


class BaseExtractor(object):
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
        if type(f_config) == str:
            self.config = yaml.load(open(f_config, 'rb'))['extractor_config']
        else:
            self.config = f_config['extractor_config']
            
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
        if ( len(ball) == 0 or
            (not ((len(np.array(ball).shape) == 3
                 and len(np.array(offense).shape) == 3
                 and len(np.array(defense).shape) == 3)
                and
                (np.array(ball).shape[1] == np.array(offense).shape[1]
                 and np.array(offense).shape[1] == np.array(defense).shape[1]))
            )
            ):
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

    def extract_batch(self, events_arr, input_is_sequence=False):
        sample_rate = 1
        Y_RANGE = 100
        X_RANGE = 50
        if input_is_sequence:
            sequences = events_arr
        else:
            sequences = np.array([make_3teams_11players(
                self.extract_raw(e)) for e in events_arr])
        # time crop (+jitter) , spatial crop
        if 'version' in self.config and self.config['version'] >= 2:
            if self.augment:
                t_jit = np.min([self.config['tfa_jitter_radius'],
                                sequences.shape[2] / 2 - self.config['tfr']])
                t_jit = (2 * t_jit * np.random.rand()
                         ).round().astype('int32') - t_jit
            else:
                t_jit = 0
            tfa = int(sequences.shape[2] / 2 + t_jit)
            sequences = sequences[:, :, tfa -
                                  self.config['tfr']:tfa + self.config['tfr']]
            if 'crop' in self.config and self.config['crop'] != '':
                reference = make_reference(sequences, self.config[
                                           'crop_size'], self.config['crop'])
                sequences = sequences - reference
                Y_RANGE = self.config['crop_size'][0] + 2
                X_RANGE = self.config['crop_size'][1] + 2
        # spatial jitter
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
        bctxy = pictorialize_fast(sequences, sample_rate, Y_RANGE, X_RANGE)

        # if cropped, shave off the extra padding
        if ('version' in self.config and self.config['version'] >= 2
                and 'crop' in self.config):
            bctxy = bctxy[:, :, :, 1:-1, 1:-1]
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



class Seq2SeqExtractor(BaseExtractor):
    """
    """

    def __init__(self, f_config):
        # super(Seq2SeqExtractor, self).__init__(f_config)    
        super(self.__class__, self).__init__(f_config)

    
    def extract_batch(self, events_arr, input_is_sequence=False):
        """
        Say, enc_time = (10) 0-10
             dec_time = (10) 11-20
             dec_target_sequence = (10) 11-20
             decoder_output = (10) 12-21
        """
        sample_rate = 1
        Y_RANGE = 100
        X_RANGE = 50
        if input_is_sequence:
            sequences = events_arr
        else:
            sequences = np.array([make_3teams_11players(
                self.extract_raw(e)) for e in events_arr])
        # spatial jitter
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
        ## temporal segment
        target_player_ind = np.random.randint(1,6)
        N_total_frames = sequences.shape[2]
        start_time =  np.round((np.random.rand() * (N_total_frames - 
                        (1+self.config['encoder_input_time']+self.config['decoder_input_time']))
                        )).astype('int32') 
        input_seq = sequences[:, :, start_time:start_time+self.config['encoder_input_time']+self.config['decoder_input_time']]
        dec_target_sequence = sequences[:, target_player_ind, start_time+self.config['encoder_input_time']
                                 :start_time+self.config['encoder_input_time']+self.config['decoder_input_time']]
        output_m1 =  sequences[:, target_player_ind, start_time+self.config['encoder_input_time'] 
                                 :1+start_time+self.config['encoder_input_time']+self.config['decoder_input_time']]
        output = output_m1[:,1:] - output_m1[:,:-1]
        ##
        bctxy = pictorialize_fast(input_seq, sample_rate, Y_RANGE, X_RANGE, keep_channels=True)        
        if self.augment and self.config['d0flip'] and np.random.rand > .5:
            bctxy = bctxy[:, :, :, ::-1]
        if self.augment and self.config['d1flip'] and np.random.rand > .5:
            bctxy = bctxy[:, :, :, :, ::-1]
        seq_inp = np.zeros((bctxy.shape[0], 4, self.config['encoder_input_time']+self.config['decoder_input_time'], Y_RANGE, X_RANGE))
        #target player
        seq_inp[:,0] = bctxy[:,target_player_ind]
        #ball
        seq_inp[:,1] = bctxy[:,0]
        #team
        seq_inp[:,2] = np.concatenate([bctxy[:,1:target_player_ind], bctxy[:,target_player_ind+1:6]], axis=1).sum(1)
        #defense
        seq_inp[:,3] = bctxy[:,6:].sum(1)
        enc_inp = seq_inp[:,:,:self.config['encoder_input_time']]
        dec_inp = seq_inp[:,:,self.config['encoder_input_time']:]
        return enc_inp, dec_inp, dec_target_sequence, output
class EncDecExtractor(BaseExtractor):
    """
    """

    def __init__(self, f_config):
        super(self.__class__, self).__init__(f_config)

    
    def extract_batch(self, events_arr, input_is_sequence=False):
        """
        Say, enc_time = (10) 0-10
             dec_time = (10) (11-10) - (20-19)
             decoder_output = (10) (12-11)-(21-20)
        """
        sample_rate = 1
        Y_RANGE = 100
        X_RANGE = 50
        if input_is_sequence:
            sequences = events_arr
        else:
            sequences = np.array([make_3teams_11players(
                self.extract_raw(e)) for e in events_arr])
        # spatial jitter
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
        ## temporal segment
        target_player_ind = np.random.randint(1,6)
        N_total_frames = sequences.shape[2]
        start_time =  1+np.round((np.random.rand() * (N_total_frames - 
                        (2+self.config['encoder_input_time']+self.config['decoder_input_time']))
                        )).astype('int32') 
        input_seq_m1 = np.array(sequences,copy=True)[:, :, start_time-1:start_time+self.config['encoder_input_time']]
        output_m1 =  np.array(sequences,copy=True)[:, target_player_ind, 
                                  -1+start_time+self.config['encoder_input_time'] 
                                 :1+start_time+self.config['encoder_input_time']+self.config['decoder_input_time']]
        output = output_m1[:,2:] - output_m1[:,1:-1]
        dec_input = output_m1[:,1:-1] - output_m1[:,:-2]
        ## Encoder Input
        if 'encoder_type' in self.config: 
            if self.config['encoder_type'] == 'target-seq':
                abs_seq = input_seq_m1[:, target_player_ind, 1:]
                abs_seq = scale_last_dim(abs_seq)
                m1_v_seq = input_seq_m1[:, target_player_ind, 1:] - input_seq_m1[:, target_player_ind, :-1]
                enc_input = np.concatenate([abs_seq, m1_v_seq], axis=-1)
                return dec_input, output, enc_input, (sequences[:, :, start_time:start_time+self.config['encoder_input_time']], target_player_ind)
                        # , sequences[:, :, start_time+self.config['encoder_input_time']:start_time+self.config['encoder_input_time']+self.config['decoder_input_time']])
            elif self.config['encoder_type'] in ['3d', '2d']:
                raise NotImplementedError()
                # bctxy = pictorialize_fast(input_seq, sample_rate, Y_RANGE, X_RANGE, keep_channels=True)        
                # if self.augment and self.config['d0flip'] and np.random.rand > .5:
                #     bctxy = bctxy[:, :, :, ::-1]
                # if self.augment and self.config['d1flip'] and np.random.rand > .5:
                #     bctxy = bctxy[:, :, :, :, ::-1]
                # seq_inp = np.zeros((bctxy.shape[0], 4, self.config['encoder_input_time']+self.config['decoder_input_time'], Y_RANGE, X_RANGE))
                # #target player
                # seq_inp[:,0] = bctxy[:,target_player_ind]
                # #ball
                # seq_inp[:,1] = bctxy[:,0]
                # #team
                # seq_inp[:,2] = np.concatenate([bctxy[:,1:target_player_ind], bctxy[:,target_player_ind+1:6]], axis=1).sum(1)
                # #defense
                # seq_inp[:,3] = bctxy[:,6:].sum(1)
                # enc_inp = seq_inp[:,:,:self.config['encoder_input_time']]
                # dec_inp = seq_inp[:,:,self.config['encoder_input_time']:]
        else: #NO encoder
            return dec_input, output, None, (sequences[:, :, start_time:start_time+self.config['encoder_input_time']], target_player_ind)
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
    from loader import BaseLoader, Seq2SeqLoader
    ##
    # f_config = 'config/train_rev0.yaml'
    # dataset = BaseDataset(f_config, 0)
    # extractor = BaseExtractor(f_config)
    # loader = BaseLoader(dataset, extractor, 35, fraction_positive=0)
    # print ('testing next_batch')
    # batch = loader.next_batch(extract=False)
    # for eind, event in enumerate(batch[0]):
    #     event.show('/home/wangkua1/Pictures/vis/%i.mp4' % eind)

    f_config = 'config/rev3-dec-single-frame.yaml'
    dataset = BaseDataset(f_config, 0)
    extractor = EncDecExtractor(f_config)
    loader = Seq2SeqLoader(dataset, extractor, 100, fraction_positive=0)
    print ('testing next_batch')
    batch = loader.next()
    # for eind, event in enumerate(batch[0]):
    #     event.show('/home/wangkua1/Pictures/vis/%i.mp4' % eind)

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
