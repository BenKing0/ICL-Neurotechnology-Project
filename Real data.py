## Importing data: (Transverse Myelitus - 1 active MU found)
import matplotlib.pyplot as plt
import numpy as np
from scipy import io
import os

dir = os.path.dirname(__file__)
os.chdir(dir)

df = io.loadmat('TM_data(1).mat')
dict_keys = df.keys() # just to print them
print(dict_keys)
emg_data = df['emg'] # (65 channels, 130672 dpoints)
activity = np.squeeze(df['activity']) # (1, 130672) squeezed to (130672,)
fs = np.squeeze(df['fSamp'])
erroneous = [np.argmax(np.max(abs(emg_data),axis=0)),np.argmax(activity)]
emg_data = np.delete(emg_data,erroneous,axis=1) # erroneous data points removed
activity = np.delete(activity,erroneous)
    
max_amp_indice = np.argmax(np.mean(emg_data,axis=1))
emg_channel = emg_data[max_amp_indice]
emg_all_channels = np.mean(emg_data,axis=0)

fig = plt.figure()
fig.add_axes((0,0,1.5,1))
plt.title('Transverse Myelitus data')
plt.ylabel('EMG signal (channel {0})'.format(max_amp_indice))
plt.plot(np.arange(len(emg_channel)),emg_channel,'k-')
plt.xticks([])
fig.add_axes((0,-1,1.5,1))
plt.plot(np.arange(len(activity)),activity,'k-')
plt.ylabel('Motor Unit Activity')
plt.xlabel('Sample number ({0} Hz freq)'.format(fs))
plt.show()

#%% Run the filtering and PE algorithms
codes_dir = 'C:/Users/44790/Documents/University/ICL Project/Papers/Codes'
os.chdir(codes_dir)
from entropies import PE, SampEn
from Denoise import Gaussian

window = 4*fs
activity_ = activity 
x = Gaussian(activity_, sigma=9, kernel_width=9).results() # recommend sigma >= kernel_width
PE_X = PE(x,bin_width=3, normalised=True, window_size=window, fixed=False).results()
xs = np.divide(np.arange(len(PE_X)) + len(activity_) - len(PE_X), fs) 

# Save results for future easy access:
np.save('cache_PE.npy',PE_X)

#%% - plot results
PE_X = np.load('cache_PE.npy')
window = 4*fs
activity_ = activity
xs = np.divide(np.arange(len(PE_X)) + len(activity_) - len(PE_X), fs) 

print('Delay of PE signal: {:.2f} sec'.format(min(xs)))
print('Signal length: {0}, PE length: {1}'.format(len(activity_),len(PE_X)))
fig = plt.figure()
fig.add_axes((0,0,1.5,1))
plt.plot(np.divide(np.arange(len(activity_)),fs),activity,'k-')
plt.ylabel('MU Activity')
plt.title('Permutation Entropy variation')   
fig.add_axes((0,-1,1.5,1))
plt.xticks(np.arange(max(xs),step=10))
plt.xlabel('Time (sec)')
plt.plot(xs,PE_X,'b--',label='Sliding window size {0}'.format(4*fs))
plt.legend(loc='lower left')
plt.ylabel('Normalised PE')
plt.show()

#%% - calculate the time bias
import adaptive_threshold as at

pred_fire = np.array(at.below(PE_X,gamma=4,window_size=1000,influence=0,disable=False))
pred_fire += len(activity_)-len(PE_X) # account for pred_fire indices starting from 0, when PE starts from a greater value
true_fire = 80000#int(len(X_one)/2)

sensitivity = 100 # number of next hits to test after each hit
to_del = []
count = 0
for i in range(len(pred_fire)-(sensitivity+1)):
    start,stop = pred_fire[i]+1,pred_fire[i]+sensitivity+1
    bool_array = [val in pred_fire for val in np.arange(start,stop)]
    if sum(bool_array) < sensitivity: # `sum(bool_array)` is number of `True`s in array - ie. `any False's?`
        to_del.append(i)
        count += 1
pred_fire = np.delete(pred_fire,to_del)

print(count)
bias = true_fire - min(pred_fire)
print('\nBias is {:.0f} samples'.format(bias))
print('Bias is {:.2f} sec'.format(bias/fs))

#%%
'''
gamma = 1
PE_threshold = np.max(PE_X) - gamma * np.std(PE_X[:int(30*window)])
PE_X_ = np.concatenate(((PE_threshold+1)*np.ones(len(activity_)-len(PE_X)),PE_X)) # fill in blanks at start with numbers greater than threshold
MU_threshold = 0.5 * max(activity_)
first_fire = np.where(activity_ > MU_threshold)[0][0]
pred_first_fire = np.where(PE_X_ < PE_threshold)[0][0]
print(PE_threshold,PE_X[0],pred_first_fire)
bias = first_fire - pred_first_fire

print('Bias is {:.0f} samples'.format(bias))
print('Bias is {:.2f} sec'.format(bias/window))


#%% Run the SampEn algorithm (a variant of it)

codes_dir = 'C:/Users/44790/Documents/University/ICL Project/Papers/Codes'
os.chdir(codes_dir)
from entropies import PE, SampEn
from Denoise import Gaussian
import tensorflow as tf
from tqdm import tqdm

size = 200
activity_ = activity
x = tf.data.Dataset.from_tensor_slices(activity_)
x = x.window(size,shift=size,drop_remainder=True)
x = x.flat_map(lambda win: win.batch(size))
SampEn_X,errors = [],[]
for bin in tqdm(x,total=len(activity_)//size):
    SE = SampEn(bin.numpy(),m=2,disable_bar=True)
    SampEn_X.append(np.nanmean(SE.results()))
    errors.append(SE.zero_error())

fig = plt.figure()
fig.add_axes((0,0,1.5,1))
plt.plot(np.arange(len(activity)),activity,'k-')
plt.ylabel('MU Activity')
plt.title('Sampe Entropy variation')
fig.add_axes((0,-1,1.5,1))
plt.xlabel(r'Window number ($T_{{rec}}$ = {0:.1f} secs)'.format(size*len(SampEn_X)/fs))
plt.plot(np.arange(len(SampEn_X)),SampEn_X,'b--',label='Each window of size {0}'.format(size))
plt.legend(loc='lower left')
plt.ylabel('Mean SampEn per window')
plt.show()

#%% Run the Lempel-Ziv algorithm
codes_dir = 'C:/Users/44790/Documents/University/ICL Project/Papers/Codes'
os.chdir(codes_dir)
from Denoise import Gaussian
import tensorflow as tf
from tqdm import tqdm
from Lempel_Ziv import LempelZiv as lz

size = 2000
activity_ = activity
complexity = lz(activity_,window_size=size,window_overlap=int(size/2),normalised=True).results()

fig = plt.figure()
fig.add_axes((0,0,1.5,1))
plt.plot(np.arange(len(activity)),activity,'k-')
plt.ylabel('MU Activity')
plt.title('LZ complexity, sliding window size = {}'.format(size))
fig.add_axes((0,-1,1.5,1))
plt.xlabel(r'Window number ($T_{{rec}}$ = {0:.1f} secs)'.format(len(activity)/fs))
plt.plot(np.arange(len(complexity)),complexity,'b--')
plt.ylabel('LZ complexity per window')
plt.show()
'''