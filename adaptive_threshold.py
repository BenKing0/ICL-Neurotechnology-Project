import numpy as np
from tqdm import trange

def below(x,gamma=2,window_size=100,influence=1,disable=False):
	trigger_indices = []
	filt_x = x.copy()
	for i in trange(window_size,len(x),disable=disable):
		prev_mean = np.mean(filt_x[i-window_size:i])
		prev_std = np.std(filt_x[i-window_size:i])
		if x[i] < (prev_mean - gamma * prev_std):
			trigger_indices.append(i)
			filt_x[i] = influence * x[i] + (1 - influence) * prev_mean#x[i-1]
	return np.array(trigger_indices,dtype=np.int)

def above(x,gamma=2,window_size=100,influence=1,disable=False):
	trigger_indices = []
	filt_x = x.copy()
	for i in trange(window_size,len(x),disable=disable):
		prev_mean = np.mean(filt_x[i-window_size:i])
		prev_std = np.std(filt_x[i-window_size:i])
		if x[i] > (prev_mean + gamma * prev_std):
			trigger_indices.append(i)
			filt_x[i] = influence * x[i] + (1 - influence) * prev_mean#x[i-1]
	return np.array(trigger_indices,dtype=np.int)