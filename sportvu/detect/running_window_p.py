from __future__ import division
import numpy as np
from utils import *


class RunWindowP:
    """	
    Given p(pnr) vs gameclock, give the start-time/end-time for PNR candidates
    type:
    	'fixed': slide window through, once a PNR detected, that window is set to zero
    				to prevent detecting multiple overlapping events
    """

    def __init__(self, config):
        self.config = config['detect_config']

    def detect(self, p, gameclock):
        cand = []  # [(start_t, end_t)...]
        if self.config['type'] == 'fixed':
            for i in xrange(self.config['window_radius'],
                            len(p) - self.config['window_radius']):
                curr_window_p = p[i - self.config['window_radius']:
                                  i + self.config['window_radius']]
                if (np.sum(curr_window_p > self.config['prob_threshold'])
                        > self.config['count_threshold']):
                    cand.append((gameclock[i - self.config['window_radius']],
                                 gameclock[i + self.config['window_radius']]))
                    curr_window_p *= 0 
        return cand
