import numpy as np

def smooth_1D_array(arr, window_radius=5, func=np.max):
	"""
	when func = np.max, it's non-maximum suppresion
	"""
	new_arr = []
	for ind, elem in enumerate(arr):
		new_arr.append(func(arr[np.max([0,ind-window_radius]):ind+window_radius]))
	return np.array(new_arr)
