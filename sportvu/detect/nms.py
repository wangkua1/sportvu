from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from utils import smooth_1D_array#, other_smooth_function


class NMS:
    """
    Non-Maximum Suppression
    """

    def __init__(self, config):
        self.config = config['detect_config']

    def detect(self, p, gameclock, return_modified_p=False):
        # # investigating change in probability
        # if self.config['prob_threshold'] > 0.8:
        #     smooth_p = smooth_1D_array(p, self.config['instance_radius'], func=np.max)
        #     plt.subplot(3, 1, 1)
        #     plt.plot(gameclock, p)
        #     plt.title('Raw Prob')
        #
        #     plt.subplot(3, 1, 2)
        #     plt.title('Smooth Prob')
        #     plt.plot(gameclock, smooth_p)
        #
        #     nms_p = smooth_p
        #     nms_p[nms_p < self.config['prob_threshold']] = -0.01
        #
        #     indices_proposal_centers = np.nonzero(nms_p == p)[0]  # local maxima
        #     cands = [gameclock[i] for i in indices_proposal_centers]
        #
        #
        #     plt.subplot(3, 1, 3)
        #     plt.title('NMS Prob')
        #     plt.plot(gameclock, nms_p)
        #     plt.plot(np.array(cands), np.ones((len(cands))), '.')
        #     plt.tight_layout()
        #     plt.show()

        smooth_p = smooth_1D_array(p, self.config['instance_radius'], func=np.max)
        nms_p = smooth_p
        nms_p[nms_p < self.config['prob_threshold']] = -0.01
        indices_proposal_centers = np.nonzero(nms_p == p)[0]  # local maxima
        cands = [(gameclock[np.max((0, i - self.config['instance_radius']))],
                  gameclock[np.min((len(gameclock) - 1, i + self.config['instance_radius']))])
                 for i in indices_proposal_centers]

        plt.close()

        if not return_modified_p:
            return cands
        else:
            return cands, nms_p, indices_proposal_centers
