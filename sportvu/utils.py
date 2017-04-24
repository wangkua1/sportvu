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