import numpy as np

def smooth_1D_array(arr, window_radius=5):
	new_arr = []
	for ind, elem in enumerate(arr):
		new_arr.append(np.mean(arr[np.max([0,ind-window_radius]):ind+window_radius]))
	return np.array(new_arr)
