import pandas as pd
from Event import Event
from Team import Team
from Constant import Constant
import cPickle as pickle



class Game:
    """A class for keeping info about the games"""
    def __init__(self, path_to_pickle, event_index):
        # self.events = None
        self.home_team = None
        self.guest_team = None
        self.event = None
        self.path_to_pickle = path_to_pickle
        self.event_index = event_index

    def read_json(self):
        # data_frame = pd.read_json(self.path_to_pickle)
        with open(self.path_to_pickle, 'rb') as handle:
            data_frame = pickle.load(handle)
        data_frame
        last_default_index = len(data_frame['events']) - 1
        print ('...')
        print (last_default_index)
        self.event_index = min(self.event_index, last_default_index)
        index = self.event_index

        print(Constant.MESSAGE + str(last_default_index))
        event = data_frame['events'][index]
        self.event = Event(event)
        self.home_team = Team(event['home']['teamid'])
        self.guest_team = Team(event['visitor']['teamid'])

    def start(self):
        self.event.show()

    def find_sequence(self, start_game_clock, end_game_clock):
        """
        find [start_game_clock, end_game_clock] sequence in Game
        return a Event with moments truncated as specified
        """
        