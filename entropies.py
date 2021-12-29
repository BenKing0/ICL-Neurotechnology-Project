import numpy as np
import itertools
from numpy import random as rnd
from tqdm import tqdm
import tensorflow as tf
import time

class PE:
    '''Return the permutation entropy (using `bin_width!` permutations) of a time series. 
    If `window size` specified then sliding window analysis is used with the given window size.'''
    def __init__(self,series,bin_width,normalised=True,window_size=None,disable_bar=False,fixed=False):
        self.x = series
        self.n = bin_width
        self.normalised = normalised
        self.window_size = window_size
        if self.window_size == None:
            self.window_size = len(self.X)+1-self.n
        self.disable = disable_bar
        self.fixed = fixed
        bin_array = self.bins()
        sequenced_bins = self.sequence(bin_array)
        motif_array,n_fact = self.permutations(sequenced_bins)
        self.permutation_entropy = self.entropy(motif_array,n_fact)
        self.motif_array = motif_array
        
    def bins(self):
        assert len(self.x) >= self.n,'Series length must be larger than bin width.'
        bin_array = []
        # Split the time-series into consecutive bins of width bin_width:
        # (Note the same can be done using TensorFlow's window method)
        t = tqdm(range(int(len(self.x)+1-self.n)),disable=self.disable)
        for i in t:
            bin_array.append(self.x[i:i+self.n])
            t.set_description(desc='Part 1',refresh=False)
        return np.array(bin_array,dtype='float64')
    
    def sequence(self,bin_array):
        sequenced_bins = []
        # Create a size-order only sequence for each bin (absolute values irrelevant):
        t = tqdm(bin_array,disable=self.disable)
        for i,bin in enumerate(t):
            # Ensures the peak_heights array is unique by incrementing jth repeat 
            # by j*min_difference/#repeats:
            difference_vector = [abs(bin[l+1]-bin[l]) for l in range(len(bin)-1)]
            try:
                min_difference = min(difference_vector[np.nonzero(difference_vector)])
            except:
                min_difference = 1 # Arbitrary as only problem for self.n = 1.
            for j,height in enumerate(bin):
                if np.where(bin==height)[0].size > 1:
                    bin[np.where(bin==height)[0]] += (np.where(bin==height)[0]*min_difference)/self.n
            sequenced_bins.append(np.argsort(bin))
            t.set_description('Part 2',refresh=False)
        sequenced_bins = np.array(sequenced_bins)
        # The higher the number in each of `self.sequenced_bins`, the bigger the height.
        return sequenced_bins
    
    def permutations(self,sequenced_bins):
        # Set the sequences from 1 to correspond neatly to permutations:
        sequence = np.add(sequenced_bins,1)
        permutations = np.array(list(itertools.permutations(np.linspace(1,self.n,self.n)))).astype(int)
        n_fact = len(permutations)
        motif_list = []
        t = tqdm(sequence,disable=self.disable)
        for i,motif in enumerate(t):
            for j,permutation in enumerate((permutations)):
                if (motif == permutation).all():
                    motif_list.append(j)
            t.set_description(desc='Part 3',refresh=False)
        motif_array = np.array(motif_list)
        assert len(motif_array) == len(sequence), 'Must have equal number of bins and motifs.'
        # A list of motif numbers corresponding to the sequence of bins
        return motif_array,n_fact
    
    def entropy(self,motif_array,n_fact):
        prob = []
        if self.fixed == True:
            assert len(motif_array) >= self.window_size,'Motif array length must not be smaller than window size.'
            start_place = int(len(motif_array)/2-self.window_size/2)
            win = motif_array[start_place:start_place+self.window_size]
            assert len(win) == self.window_size
            prob = [len(np.where(win==i)[0])/(len(win)) for i in range(n_fact)]
            prob = np.array(prob)
            assert np.isclose(np.sum(prob),1), 'Sum of probabilities must equal 1 at every time step'
            # The elements of probability = 0 is set to 1, as lim(p->0)[H(0)]=H(1)=0, but H(0) undefined.
            prob[np.where(prob==0)[0]] = 1
            entropy = np.sum(-prob*np.log2(prob))
        else:
            # Implements a sliding window of width `window_size` using TensorFlow:
            assert len(motif_array) >= self.window_size,'Motif array length must not be smaller than window size.'
            dataset = tf.data.Dataset.from_tensor_slices(motif_array)
            dataset = dataset.window(self.window_size,shift=1,drop_remainder=True)
            dataset = dataset.flat_map(lambda window: window.batch(self.window_size))
            t = tqdm(dataset,total=int(len(motif_array)+1-self.window_size),disable=self.disable)
            for window in t:
                window = window.numpy()
                prob_per_motif = ([len(np.where(window==i)[0])/(len(window)) for i in range(n_fact)])
                prob.append(prob_per_motif)
                t.set_description(desc='Part 4')
            # `prob` now contains the probability for each of the n! motifs taken using each window:
            prob = np.array(prob)
            for j in (range(len(prob))):
                assert np.isclose(np.sum(prob[j]),1), 'Sum of probabilities must equal 1 at every time step'
                # For mathematical reasons - undefined entropy for p = 0; H(1)=H(0)= 0:
                prob[j][np.where(prob[j]==0)[0]] = 1
            entropy = np.sum(-prob*np.log2(prob),axis=1)  
        self.n_fact = n_fact
        return entropy
    
    def results(self):
        if self.normalised == False:
            permutation_entropy = self.permutation_entropy
        else:
            # Normalised to log_2(n!)
            permutation_entropy = self.permutation_entropy/np.log2(self.n_fact)
        return permutation_entropy
    
    def motifs(self):
        return self.motif_array
    
class SampEn:
    '''Return the sample entropy (SINGLE-VALUE FOR EACH MOTIF) of a time-series using the
    `Costa et al.` method.'''
    def __init__(self,x,m=3,r=0.2,tau=1,distance='inf',disable_bar=False,use_own=True):
        # `r` is multiple of data std deviation in window. `m` is smallest embedding dimension.
        # `distance` is to define the distance method using L_p distance. `inf` is Chebyshev.
        self.disable = disable_bar
        self.SampEn_array = self.entropy(x,m,r,distance,tau,use_own)
     
    def entropy(self,x,m,r,distance,tau,use_own):
        if use_own == True:
            # Set-up sliding windows:
            tol = r*np.std(x)
            begin = time.perf_counter()
            x = tf.data.Dataset.from_tensor_slices(x)
            xmi = x.window(m,shift=tau,drop_remainder=True).flat_map(lambda f: f.batch(m))
            xmj = x.window(m+1,shift=tau,drop_remainder=True).flat_map(lambda f: f.batch(m+1))
            xmi = [element.numpy() for element in xmi]
            xmj = [element.numpy() for element in xmj]
            end = time.perf_counter()
            if self.disable == False:
                print('Time to batch: {:.2f}'.format(end-begin))
            SampEn_vctr = []
            begin = time.perf_counter()
            t = tqdm(range(len(xmj)),disable=self.disable)
            self.error = 0
            for i in t:
                # this is the slow part
                d = [self.distance([xmi[i],comp],distance) for comp in xmi]
                B = len(np.where(d<=tol)[0])-1
                d = [self.distance([xmj[i],comp],distance) for comp in xmj]
                A = len(np.where(d<=tol)[0])-1
                assert A <= B, 'Logic error.'
                if A == 0 and B == 0:
                    SampEn_vctr.append(0)
                elif A == 0:
                    try:
                        SampEn_vctr.append(SampEn_vctr[-1])
                    except:
                        self.error += 1
                        SampEn_vctr.append(np.nan)
                else:
                    SampEn_vctr.append(-np.log(A/B))
            end = time.perf_counter()
            if self.disable == False:
                print('Time to calculate: {:.2f}'.format(end-begin))
                print('Divide by 0 error: {}'.format(self.error))
        else:
            begin = time.perf_counter()
            SampEn_vctr = []
            x = tf.data.Dataset.from_tensor_slices(x)
            x = x.window(100,shift=tau,drop_remainder=True).flat_map(lambda f: f.batch(m))
            for w in x:
                w = w.numpy()
                SampEn_vctr.append(self.use_sampEn_wiki(w,m,r))
            end = time.perf_counter()
            print('Time to calculate: {:.2f}'.format(end-begin))
        SampEn_array = np.array(SampEn_vctr)
        return SampEn_array
    
    def distance(self,vectors,distance):
        # `vectors` of type [vctr1, vctr2].
        if distance == 'inf':
            # Return Chebyshev (L_inf) distance:
            dist = abs(np.subtract(vectors[0],vectors[1])).max()
        else:
            try:
                p = int(distance)
                assert p > 0, 'Distace must be a positive integer (or `inf`).'
            except:
                print('Distace must be a positive integer or `inf`.')
            dist = np.power(np.sum(np.power(abs(np.subtract(vectors[0],vectors[1])),p)),1/p)
        return dist
    
    def results(self):
        return self.SampEn_array
    
    def zero_error(self):
        return self.error
    
    def use_sampEn_wiki(self,L, m, r): # (Stolen from Wikipedia page)
        N = len(L)
        B = 0.0
        A = 0.0
        # Split time series and save all templates of length m
        xmi = np.array([L[i : i + m] for i in range(N - m)])
        xmj = np.array([L[i : i + m] for i in range(N - m + 1)])
        # Save all matches minus the self-match, compute B
        B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])
        # Similar for computing A
        m += 1
        xm = np.array([L[i : i + m] for i in range(N - m + 1)])
        A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])
        assert A <= B
        if A != 0:
            return -np.log(A / B)
        if B == 0:
            assert A == 0
            return 0
        if A == 0 and B != 0:
            return 1
    
class MSE:
    '''Multiscale entropy: Prepare the dataset with `scale` course-grained time series. 
    The PE function can then be used to calculate the permutation entropy for each scale
    to find the natural time resolution of the time-series.'''
    def __init__(self,x,scales=[1],compute_PE=False,compute_SE=False,return_means=False,**kwargs):
        series = self.build_series(x,scales)
        self.scales = scales
        self.return_means = return_means
        assert len(series) == len(scales), '`scales` array length mismatch.'
        if compute_PE == True:
            self.entropies = self.Perm_entropy(series,kwargs)
            
        elif compute_SE == True:
            self.entropies = self.Samp_entropy(series,kwargs)
        else:
            self.return_means = False
            #! not actually true but it returns the desired result (the course-grained series):
            self.entropies = series
        if self.return_means == True:
            self.means = [np.mean(self.entropies[i]) for i in range(len(scales))]
            self.stds = [np.std(self.entropies[i]) for i in range(len(scales))]
        
    def build_series(self,x,scales):
        length = len(x)
        x = tf.data.Dataset.from_tensor_slices(x)
        series = []
        t = tqdm(scales)
        for i,scale in enumerate(t):
            t.set_description(desc='Re-scale')
            X = x.window(scale,shift=scale,drop_remainder=True)
            X = X.flat_map(lambda window: window.batch(scale))
            y = [np.mean(i.numpy()) for i in X]
            assert int(len(y)) == int(length/scale), 'Ensuring series sizes match.'
            y = np.array(y)
            series.append(y)
        return np.array(series)
    
    def Perm_entropy(self,series,kwargs):
        bin_width = 3
        normalised = True
        window_size_set = 1000
        for key,value in kwargs.items():
            if key == 'window_size':
                window_size_set = value
            if key == 'bin_width':
                bin_width = value
            if key == 'normalised':
                normalised = value
        t = tqdm(series)
        entropy = []
        for i,x in enumerate(t):
            window_size = window_size_set 
            if window_size > (len(x)+1-bin_width):
                window_size = int(len(x)+1-bin_width)
                changed = True
            else:
                changed = False
            t.set_description(desc='Calcluate PE: scale = {}'.format(self.scales[i]))
            entropy.append(PE(x,bin_width=bin_width,normalised=normalised,window_size=window_size,disable_bar=True).results())   
        if changed == True:
            print('Window size was too large and was changed to its maximum allowed value')
        return np.array(entropy)
    
    def Samp_entropy(self,series,kwargs):
        m = 3
        r = 0.2
        tau = 1
        distance = 'inf'
        for key,value in kwargs.items():
            if key == 'm':
                m = value
            if key == 'r':
                r = value
            if key == 'tau':
                tau = value
            if key == 'distance':
                distance == value
        t = tqdm(series)
        entropy = []
        for i,x in enumerate(t):
            t.set_description(desc='Calcluate SE: scale = {}'.format(self.scales[i]))
            entropy.append(SampEn(x,m=m,r=r,tau=tau,distance=distance,disable_bar=True).results())   
        return np.array(entropy)
    
    def results(self):
        if self.return_means == True:
            return self.means,self.stds
        else:
            return self.entropies
        
#%% - Permutation entropy
'''The PE function is inefficient for large bin-sizes due to many 
iterations over the values in the bins, and a large value of (self.n)! - 
however, the bins have to be large enough to get meaningul patterns in them.

The sliding window entropy calculation is only really suitable when the 
length of the time series is a lot greater than the chosen window length.

Window length must be chosen to be large enough for probabilities to be 
meaningully calculated, but not so large as to have the addition of an unusual 
motif make negligible difference to the probability distribution (as is the 
issue if a sliding window is not used for lengthy time series).
Smaller window sizes also greatly reduce computation time due to iterations.'''