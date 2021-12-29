import numpy as np
import tensorflow as tf

class sumSquares:
    def __init__(self,x,lag=2):
        self.x,self.lag = x,lag
        batched = tf.data.Dataset.from_tensor_slices(self.x)
        batched = batched.window(size=lag,shift=lag,drop_remainder=True)
        self.batched = batched.flat_map(lambda f: f.batch(lag,drop_remainder=True))
        self.sumSquaresOP = [np.sum(np.power(vctr.numpy(),2)) for vctr in self.batched]
    
    def __call__(self):
        return np.array(self.sumSquaresOP,dtype=np.double)
        
    
class globalThreshold(sumSquares): # `threshold` class now inherits the `self` variables from `sumSquares`
    '''This class doesn't batch the input - if adaptive threshold is 
    wanted with a sliding window then do that seperately.'''
    
    def __init__(self,x,threshold,lag=2,direction='above'):
        super().__init__(x,lag) # passes `x` and `lag` into `sumSquares` if only sumSquares is called
        self.z,self.threshold = np.array(self.sumSquaresOP,dtype=np.double),threshold
        assert all(self.x) >= 0
        if direction.lower() == 'above':
            self.indices = np.where(self.z > threshold)[0] # indices of z that are above threshold, each represents 2 of PE's
        elif direction.lower() == 'below':
            self.indices = np.where(self.z < threshold)[0]   
            
    def __call__(self):
        return np.array(self.indices,dtype=np.int),self.z
    
    
# Do this by using the chi-squared probability distribution to find a 
# threshold after which the probability of false detection is less than 0.05
'''class adaptiveThreshold(sumSquares):
    def __init__(self,x,lag=2,window_size=10,direction='above'):
        super().__init__(x,lag)
        self.z = np.array(self.sumSquaresOP,dtype=np.double)
        batched = tf.data.Dataset.from_tensor_slices(self.z)
        batched = batched.window(window_size,lag,drop_remainder=True)
        batched = batched.flat_map(lambda f: f.batch(window_size))'''
        