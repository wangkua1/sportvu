from __future__ import division
import numpy as np
from utils import *
import math
import matplotlib.pyplot as plt

class RunWindowP:
    """
    Given p(pnr) vs gameclock, give the start-time/end-time for PNR candidates
    type:
    	'fixed': slide window through, once a PNR detected, that window is set to zero
    				to prevent detecting multiple overlapping events
    """

    def __init__(self, config):
        self.config = config['detect_config']

    def detect(self, p, gameclock, return_modified_p=False):
        cand = []  # [(start_t, end_t)...]
        indices_proposal_centers = []

        # # investigating change in probability
        # if self.config['prob_threshold'] > 0.8:
        #     plt.subplot(2, 1, 1)
        #     plt.plot(gameclock, p)
        #     plt.title('Raw Prob')
        #
        #     plt.subplot(2, 1, 2)
        #     plt.title('Count Thresh At Prob Thresh')
        #     plt.plot(gameclock, np.ones((len(gameclock))) * self.config['prob_threshold'], '-')
        #     plt.plot(gameclock, p)
        #
        #     if self.config['type'] == 'fixed':
        #         for i in xrange(self.config['window_radius'],
        #                         len(p) - self.config['window_radius']):
        #             curr_window_p = p[i - self.config['window_radius']:
        #             i + self.config['window_radius']]
        #             if (np.sum(curr_window_p > self.config['prob_threshold'])
        #                     > self.config['count_threshold']):
        #                 cand.append((gameclock[i - self.config['window_radius']],
        #                              gameclock[i + self.config['window_radius']]))
        #                 indices_proposal_centers.append(
        #                     int(i + math.ceil(self.config['window_radius'] / 2)))  # local maxima
        #                 curr_window_p *= 0
        #
        #     cands = [gameclock[i] for i in indices_proposal_centers]
        #     plt.plot(np.array(cands), np.ones((len(cands))), '.')
        #     plt.tight_layout()
        #     plt.show()
        #     plt.close()

        if self.config['type'] == 'fixed':
            for i in xrange(self.config['window_radius'],
                            len(p) - self.config['window_radius']):
                curr_window_p = p[i - self.config['window_radius']:
                                  i + self.config['window_radius']]
                if (np.sum(curr_window_p > self.config['prob_threshold'])
                        > self.config['count_threshold']):
                    cand.append((gameclock[i - self.config['window_radius']],
                                 gameclock[i + self.config['window_radius']]))
                    indices_proposal_centers.append(int(i + math.ceil(self.config['window_radius']/2)))  # local maxima
                    curr_window_p *= 0
        if not return_modified_p:
            return cand
        else:
            return cand, None, indices_proposal_centers
