from scipy.signal import find_peaks
import numpy as np
import tensorflow as tf
from operator import itemgetter
from tqdm import tqdm, trange

class Gaussian:
    def __init__(self,x,sigma=1,kernel_width=5):
        self.weighted_mean_array = self.rolling_mean(x,kernel_width,sigma)
        
    def rolling_mean(self,x,kernel_width,sigma):
        dataset = tf.data.Dataset.from_tensor_slices(x)
        dataset = dataset.window(kernel_width,shift=1,drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(kernel_width))
        weighted_mean = []
        for kernel in tqdm(dataset,total=len(x)+1-kernel_width):
            kernel = kernel.numpy()
            centre = int(np.ceil(len(kernel)/2))-1 # `-1` to account for 0 index
            weights = self.dist(np.subtract(np.arange(kernel_width),centre),sigma)
            weights = weights*(1/np.sum(weights))
            assert np.isclose(np.sum(weights),1), 'Sum of weights must equal 1.'
            mean_value = np.sum(np.multiply(kernel,weights))
            weighted_mean.append(mean_value)
        return np.array(weighted_mean)
    
    def dist(self,x,sigma):
        return (1/np.sqrt(2*np.pi*sigma**2))*np.exp(-0.5*(x/sigma)**2)
    
    def results(self):
        return self.weighted_mean_array
        
class exp_smoothing:
    def __init__(self,x,args,order=1,kernel_width=5):
        #assert args[np.arange(len(args))]<=1 and args[np.arange(len(args))]>=0, 'Arguments must all be between 0 and 1.' 
        dataset = tf.data.Dataset.from_tensor_slices(x)
        dataset = dataset.window(kernel_width,shift=1,drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(kernel_width))
        if order == 1:
            alpha = itemgetter(0)(args) 
            self.output = self.first_order(dataset,alpha)
        if order == 2:
            alpha,beta = itemgetter(0,1)(args) 
            self.output = self.second_order(dataset,alpha,beta)
        if order == 3:
            alpha,beta,gamma = itemgetter(0,1,2)(args) 
            self.output = self.third_order(dataset,alpha,beta,gamma)
    
    def first_order(self,dataset,alpha):
        # `alpha` defines the extent of smoothing - Overall smoothing.
        smoothed_array = []
        for kernel in dataset:
            kernel = kernel.numpy()
            smoothed = np.zeros(kernel.shape[0])
            smoothed[0] = kernel[0]
            for i in range(1,len(kernel)):
                 smoothed[i] = kernel[i]*alpha+(1-alpha)*smoothed[i-1]
            smoothed_array.append(smoothed[-1])                                                                        
        return np.array(smoothed_array)
    
    def second_order(self,dataset,alpha,beta):
        # `beta` accounts for the slope - Trend smoothing.
        smoothed_array = []
        diff_array = []
        for kernel in dataset:
            kernel = kernel.numpy()
            smoothed = np.zeros(kernel.shape[0])
            diff = np.zeros(int(kernel.shape[0]-1))
            smoothed[0:2] = kernel[0:2]
            diff[0] = kernel[1]-kernel[0]
            for i in range(2,len(kernel)):
                smoothed[i] = kernel[i]*alpha+(1-alpha)*(smoothed[i-1]+diff[i-1])
                diff[i-1] = beta*(smoothed[i-1]-smoothed[i-2])+(1-beta)*diff[i-2]
            smoothed_array.append(smoothed[-1])
            diff_array.append(diff[-1])
        return np.array(smoothed_array),np.array(diff_array)
    
    def third_order(self,dataset,alpha,beta,gamma):
        # `gamma` dictates the seasonality - Seasonality smoothing.
        for kernel in dataset:
            kernel = kernel.numpy()
        return np.array(smoothed_array),np.array(diff_array),np.array(corr_factors)
    
    def results(self):
        return self.output
    
class reconstruct_peaks:
    def __init__(self,x,height=0,threshold=None,distance=None):
       indices,heights  = self.pick_peaks(x,height=height,threshold=threshold,
                                          distance=distance)
       self.spike_train = self.build_series(x,indices,heights)
        
    def pick_peaks(self,x,height,threshold,distance):
       peak_indices,H = find_peaks(x,height=height,threshold=threshold,
                                          distance=distance)
       heights = H['peak_heights']
       return peak_indices,heights
   
    def build_series(self,x,indices,heights):
        spike_train = np.zeros(x.shape)
        spike_train[indices] = heights
        return spike_train
    
    def results(self):
        return self.spike_train
        
#%%