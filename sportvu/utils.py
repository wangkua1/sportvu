import numpy as np

def truncated_mean(s, cut_fraction=.2):
    s = np.sort(np.array(s, copy=True))
    N = len(s)
    cut = int(np.round(N*cut_fraction))
    return np.mean(s[cut:-cut])

def experpolate_position(start_frame, pred):
    """
    start_frame is (B,2) of unnormalized absolute (x,y)
    pred is (B,T,2) of unnormalized d(x,y)
    """
    shape = list(pred.shape)
    shape[1] +=1
    ret_val = np.zeros(shape)
    ret_val[:,0] = start_frame
    # ret_val[:,0,0] *= 100
    # ret_val[:,0,1] *= 50
    for t in xrange(pred.shape[1]):
        ret_val[:,t+1] =  ret_val[:,t] + pred[:,t]
    return ret_val

def find_discontinuous_sequences(seqs, mean_diff_thresh=2.):
    """
    Given (N,11,T,2), return (N,) boolean array
        where True means it's a discontinuous sequence
    """
    diff = seqs[:,:,1:] - seqs[:,:,:-1] #(N,11,T-1,2)
    team_diff_vs_time = np.abs(
            np.concatenate(list(np.transpose(diff, (1,0,2,3))), axis=-1)
                            ).mean(-1) #(N,T)
    return np.max(team_diff_vs_time, axis=-1) > mean_diff_thresh

def filter_discontinuous(seqs, mean_diff_thresh=2.):
    """
    Given (N,11,T,2), return (N-k,11,T,2) seqs
        where k is the number of discontinuous sequences
    """
    inds = find_discontinuous_sequences(seqs, mean_diff_thresh)
    arr = np.where(inds==False)[0]
    if len(arr) == 0:
        return seqs
    else:
        return seqs[arr]

def dist_trajectory(y,y_,keep_first=False):
    """
    both numpy arrays with shape (N, T, 2*#players)
        return sum_(time, players) L2_dist
    """
    def _reshape(y):
        return np.reshape(y, (y.shape[0],y.shape[1], -1, 2))
    y = _reshape(y)
    y_ = _reshape(y_)
    if keep_first:
        return np.power(np.sum(np.power((y-y_), 2),-1),.5).mean(-1).mean(-1)
    else:
        return np.power(np.sum(np.power((y-y_), 2),-1),.5).mean()