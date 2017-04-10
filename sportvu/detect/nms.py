from __future__ import division
import numpy as np
from utils import smooth_1D_array


class NMS:
    """ 
    Non-Maximum Suppression
    """

    def __init__(self, config):
        self.config = config['detect_config']

    def detect(self, p, gameclock, return_modified_p=False):
        nms_p = smooth_1D_array(p, self.config['window_radius'], func=np.max)
        nms_p[nms_p < self.config['prob_threshold']] = -0.01
        indices_proposal_centers = np.nonzero(nms_p == p)[0]  # local maxima
        cands = [(gameclock[np.max((0, i - self.config['window_radius']))],
                  gameclock[np.min((len(gameclock) - 1, i + self.config['window_radius']))])
                 for i in indices_proposal_centers]
        if not return_modified_p:
            return cands
        else:
            return cands, nms_p
