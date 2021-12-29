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

#%% START GAUSSIAN

import Denoise as dn
X = dn.Gaussian(activity,sigma=9,kernel_width=9).results()
plt.plot(np.arange(len(X))/fs,X)
plt.show()

#%%
import entropies

winsize = 4*fs
PE_X = entropies.PE(X,bin_width=3,normalised=True,window_size=winsize).results()
np.save('C:/Users/44790/Documents/University/ICL Project/Papers/Codes/Cache/TransMyl_{}.npy'.format(str(winsize)),PE_X)

#%%

import numpy as np
winsize = 4*fs
PE_X = np.load('C:/Users/44790/Documents/University/ICL Project/Papers/Codes/Cache/TransMyl_{}.npy'.format(str(winsize)))
shift = len(X) - len(PE_X)
xs = np.divide((np.arange(len(PE_X)) + shift),fs)
fig = plt.figure()
fig.add_axes((0,0,1.5,0.65))
plt.title('MU activity') 
plt.plot(np.arange(len(X)),X)
plt.ylabel('Signal (mV) - filtered')
plt.xticks([])
fig.add_axes((0,-0.9,1.5,0.85))
plt.plot(xs,PE_X)
plt.ylabel('PE (normalised)')
plt.xlabel('Time (sec)')
plt.xticks(np.arange(max(xs),step=3))
plt.show()
 
#%% Bonato

from Bonato import sumSquares as ss
from Bonato import globalThreshold as gt
from tqdm import trange

'''noisey = np.random.normal(size=3000)
gf_noisey = dn.Gaussian(noisey,kernel_width=3,sigma=9).results()
PE_noise = entropies.PE(gf_noisey,bin_width=3,normalised=True,window_size=2048).results()
mu = np.mean(PE_noise)
sigma = np.std(PE_noise)'''

mu = np.mean(PE_X[:fs])
sigma = np.std(PE_X[:fs])

print('\nNoise mean: {0:.3e}, Noise standard dev: {1:.3e}'.format(mu,sigma))

z_stat_signal = (PE_X - mu) / sigma 
zeta = 20

result = gt(z_stat_signal,lag=2,threshold=zeta,direction='above')() # `above`, as sum of squares v large for both v positive AND v negative z_statistics
indices = result[0]
print('\nNumber of Highlighted indices: {}'.format(len(indices)))
print('\nSum Square values: ')

sensitivity = fs//10 # number of next hits to test after each hit
to_del = []
count = 0
from tqdm import trange
for i in trange(len(indices)-(sensitivity+1)):
    start,stop = indices[i]+1,indices[i]+sensitivity+1
    bool_array = [val in indices for val in np.arange(start,stop)]
    if sum(bool_array) < sensitivity-1: # `sum(bool_array)` is number of `True`s in array - ie. `any False's?`
        to_del.append(i)
        count += 1
    if PE_X[2*indices[i]] >= mu:
        to_del.append(i)
        count += 1
indices = np.delete(indices,to_del)
print('\nNumber of removed indices: {}'.format(count))
indices = 2 * indices

xs = np.divide((np.arange(len(PE_X)) + shift),fs)
fig = plt.figure()
fig.add_axes((0,0,1.5,0.65))
plt.title('Transverse Mylitus signal')
plt.plot(np.arange(len(X)),X)
plt.ylabel('Signal (mV) - filtered')
plt.xticks([])
fig.add_axes((0,-0.9,1.5,0.85))
plt.plot(xs,PE_X)
plt.ylabel('PE (normalised)')
plt.xlabel('Time (sec)')
plt.plot((indices+shift)/fs,PE_X[indices],'r.')
plt.xticks(np.arange(max(xs),step=3))
plt.show()

'''
pred_fire = indices[0]
true_fire = int(len(X_one)/2)
bias = true_fire - pred_fire - shift
print('\nBias is {:.0f} samples'.format(bias))
print('Bias is {:.2f} sec'.format(bias/fs))
'''