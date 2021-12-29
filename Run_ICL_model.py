#%% - Bar of mean PE for guassian/exponential/butterworth filters vs non-filtered
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
import readers 
import Denoise as dn
import entropies as PE
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt

fs = 2048
N_MU = 10
SNR = 30
T = 5

X = readers.GetSimData([2,1,1000,2000])[0]
ind = np.argmax(np.max(X,axis=0))

'''Set-up the Butterworth filter for band-pass between lb and ub frequencies.'''
nyq = fs/2
lb = 20
ub = 500
order = 4
b,a = butter(order,(lb,ub)/np.array(nyq),btype='bandpass')

'''`args` of form [sample time, number of MUs, SNR, sampling freq] - 
ie. 2048 Hz sampling frquency means 2048 samples for every sample time value.'''

PE_X,PE_N,PE_X_filt,PE_N_filt,PE_X_E,PE_N_E,PE_X_B,PE_N_B = [],[],[],[],[],[],[],[]
for repeats in np.arange(10):   
    args = [T,N_MU,SNR,fs]
    X = readers.GetSimData(args)[0]
    X = X.T[ind]
    PE_X.append(PE.PE(X,3,True,window_size=fs,fixed=True).results())
    X_G = dn.Gaussian(X,sigma=3,kernel_width=9).results()
    PE_X_filt.append(PE.PE(X_G,3,True,window_size=fs,fixed=True).results())
    X_E = dn.exp_smoothing(X,[0.4],order=1,kernel_width=9).results()
    PE_X_E.append(PE.PE(X_E,3,True,window_size=fs,fixed=True).results())
    X_B = filtfilt(b,a,X+(1e-5),axis=0)
    PE_X_B.append(PE.PE(X_B,3,True,window_size=fs,fixed=True).results())
    
    N = np.random.normal(0,np.std(X),X.shape[0])
    PE_N.append(PE.PE(N,3,True,window_size=fs,fixed=True).results())
    N_G = dn.Gaussian(N,sigma=3,kernel_width=9).results()
    PE_N_filt.append(PE.PE(N_G,3,True,window_size=fs,fixed=True).results())
    N_E = dn.exp_smoothing(N,[0.4],order=1,kernel_width=9).results()
    PE_N_E.append(PE.PE(N_E,3,True,window_size=fs,fixed=True).results())
    N_B = filtfilt(b,a,N+(1e-5),axis=0)
    PE_N_B.append(PE.PE(N_B,3,True,window_size=fs,fixed=True).results())
    print('repeat {} done'.format(repeats))
    
mean_filt_XE = np.mean(PE_X_E)
std_filt_XE = np.std(PE_X_E)
mean_filt_NE = np.mean(PE_N_E)
std_filt_NE = np.std(PE_N_E)

mean_filt_XB = np.mean(PE_X_B)
std_filt_XB = np.std(PE_X_B)
mean_filt_NB = np.mean(PE_N_B)
std_filt_NB = np.std(PE_N_B)

mean_filt_XG = np.mean(PE_X_filt)
std_filt_XG = np.std(PE_X_filt)
mean_filt_NG = np.mean(PE_N_filt)
std_filt_NG = np.std(PE_N_filt)

mean_X = np.mean(PE_X)
std_X = np.std(PE_X)
mean_N = np.mean(PE_N)
std_N = np.std(PE_N)

mean_X = [mean_X,mean_filt_XG,mean_filt_XE,mean_filt_XB]
mean_N = [mean_N,mean_filt_NG,mean_filt_NE,mean_filt_NB]
std_X = [std_X,std_filt_XG,std_filt_XE,std_filt_XB]
std_N = [std_N,std_filt_NG,std_filt_NE,std_filt_NB]

labels = ['None','Guassian','Exponential','Butterworth']
x = np.arange(len(labels))  # the label locations
width = 0.32  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, mean_X, yerr=std_X, width=width, label='{} MUs'.format(N_MU))
rects2 = ax.bar(x + width/2, mean_N, yerr=std_N, width=width, label='Noise') 
ax.set_ylabel('Mean PE (normalised)')
ax.set_xlabel('Filter type')
ax.set_title('PE variation with Filter Type')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='lower left')

fig.tight_layout()
plt.show()

#%% - Examine how PE varies with sampling frequency: 256-8192 Hz.
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
import readers 
import Denoise as dn
import entropies as PE
import numpy as np
import matplotlib.pyplot as plt

N_MU = 10
SNR = 30
T = 5
fs = [256,512,1024,2048,4096,8192]

X = readers.GetSimData([2,1,1000,2000])[0]
ind = np.argmax(np.max(X,axis=0))

def generate_series(x,SNR,reference=None):
    mean = np.mean(x)
    stddev = (np.std(x)/(10**(SNR/20)))  
    if len(reference) != 0:
        mean = np.mean(reference)
        stddev = (np.std(reference)/(10**(SNR/20)))  
    noise = np.random.normal(mean,stddev,len(x))
    pattern = np.add(x,noise)
    return pattern;

repeats = 10
mean_X,mean_N,std_X,std_N = [],[],[],[]
for N_MU in [0,2,10]:
    PE_X,PE_N = [],[]
    for freq in fs:
        for repeat in np.arange(repeats):
            if N_MU != 0:
                args = [T,N_MU,SNR,freq]
                X = readers.GetSimData(args)[0]
                X = X.T[ind]
                X = dn.Gaussian(X,sigma=3,kernel_width=9).results()
                PE_X.append(PE.PE(X,3,window_size=freq,fixed=True).results())
            if N_MU == 0:
                N = generate_series(np.linspace(0,0,freq*T),10,reference=X)
                N = dn.Gaussian(N,sigma=3,kernel_width=9).results()
                PE_N.append(PE.PE(N,3,window_size=freq,fixed=True).results())
        if N_MU != 0:
            mean_X.append(np.mean(PE_X))
            std_X.append(np.std(PE_X))
        if N_MU == 0:
            mean_N.append(np.mean(PE_N))
            std_N.append(np.std(PE_N))
        print('freq {0}, MU number {1} done'.format(freq,N_MU))

labels = [256,512,1024,2048,4096,8192]#fs
x = np.arange(len(labels))  # the label locations
width = 0.27  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width, mean_X[:6], yerr=std_X[:6], width=width, label='{} MUs'.format(2),color='blue')
rects2 = ax.bar(x, mean_X[6:], yerr=std_X[6:], width=width, label='{} MUs'.format(10),color='orange')
rects3 = ax.bar(x + 3*width/3, mean_N, yerr=std_N, width=width, label='Noise',color='red') 
ax.set_ylabel('Mean PE (normalised)')
ax.set_xlabel('Sampling frequency (Hz)')
ax.set_title('PE variation with sampling frequency - {} repeats per fs'.format(repeats))
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='lower left')

fig.tight_layout()
plt.show()

#%% - Examine how PE varies with PE sliding window size: [0.2,0.4,0.6,1,2,4]*fs.
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
import readers 
import Denoise as dn
import entropies as PE
import numpy as np
import matplotlib.pyplot as plt

N_MU = 10
SNR = 30
T = 5
fs = 2048
window = (np.array([0.2,0.4,0.6,1,2,4])*fs).astype(int)
Xi = readers.GetSimData([2,1,1000,2000])[0]
ind = np.argmax(np.max(Xi,axis=0))

def generate_series(x,SNR,reference=None):
    mean = np.mean(x)
    stddev = (np.std(x)/(10**(SNR/20)))  
    if len(reference) != 0:
        mean = np.mean(reference)
        stddev = (np.std(reference)/(10**(SNR/20)))  
    noise = np.random.normal(mean,stddev,len(x))
    pattern = np.add(x,noise)
    return pattern;

repeats = 10
std_X,std_N,mean_X,mean_N = [],[],[],[]
for N_MU in [0,2,10]:
    for size in window:
        PE_X,PE_N = [],[]
        for repeat in np.arange(repeats):
            if N_MU == 0:
                N = generate_series(np.linspace(0,0,fs*T), SNR, reference=Xi)
                N = dn.Gaussian(N,sigma=3,kernel_width=9).results()
                PE_N.append(PE.PE(N,3,window_size=size,fixed=True).results())
            else:
                args = [T,N_MU,SNR,fs]
                X = readers.GetSimData(args)[0]
                X = X.T[ind]
                X = dn.Gaussian(X,sigma=3,kernel_width=9).results()
                PE_X.append(PE.PE(X,3,window_size=size,fixed=True).results())
        if N_MU == 0:
            mean_N.append(np.mean(PE_N))
            std_N.append(np.std(PE_N))
        else:
            mean_X.append(np.mean(PE_X))
            std_X.append(np.std(PE_X))
        print('Size {0}, MU number {1} done'.format(size,N_MU))

labels = window
x = np.arange(len(labels))  # the label locations
width = 0.27  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width, mean_X[:6], yerr=std_X[:6], width=width, label='{} MUs'.format(2),color='blue')
rects2 = ax.bar(x, mean_X[6:], yerr=std_X[6:], width=width, label='{} MUs'.format(10),color='orange')
rects3 = ax.bar(x + 3*width/3, mean_N, yerr=std_N, width=width, label='Noise',color='red') 
ax.set_ylabel('Mean PE (normalised)')
ax.set_xlabel('Window size')
ax.set_title('PE variation with sliding window size - {} repeats'.format(repeats))
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='lower left')

fig.tight_layout()
plt.show()

#%% - Examine how PE varies with # MUs: [1,5,10,15].
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
import readers 
import Denoise as dn
import entropies as PE
import numpy as np
import matplotlib.pyplot as plt

N_MU = [0,1,2,3,4,5,10,20,40]
SNR = 30
T = 5
fs = 2048
window = 2048

Xi = readers.GetSimData([2,1,1000,2000])[0]
ind = np.argmax(np.max(Xi,axis=0))

def generate_series(x,SNR,reference=[]):
    mean = np.mean(x)
    stddev = (np.std(x)/(10**(SNR/20)))  
    if len(reference) != 0:
        mean = np.mean(reference)
        stddev = (np.std(reference)/(10**(SNR/20)))  
    noise = np.random.normal(mean,stddev,len(x))
    pattern = np.add(x,noise)
    return pattern;

repeats = 10
means,stds = [],[]
for SNR in [10,30]:
    for num in N_MU:
        PEnt = []
        for repeat in np.arange(repeats):
            if num == 0:
                N = generate_series(np.linspace(0,0,len(X)),SNR,reference=Xi)
                N = dn.Gaussian(N,sigma=3,kernel_width=9).results()
                PEnt.append(PE.PE(N,3,window_size=window,fixed=True).results())
            else:
                args = [T,num,SNR,fs]
                X = readers.GetSimData(args)[0].T 
                X = X[ind]
                X = dn.Gaussian(X,sigma=3,kernel_width=9).results()
                PEnt.append(PE.PE(X,3,window_size=window,fixed=True).results())
        means.append(np.mean(PEnt))
        stds.append(np.std(PEnt))
        print('Num. {} done'.format(num))
    print('SNR {} done'.format(SNR))
means = means/means[0]
stds = stds/means[0]**2

labels = N_MU
x = np.arange(len(labels))  # the label locations
width = 0.4  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x-width/2, means[9:], yerr=stds[9:],width=width,label='SNR: {} dB'.format(30))
rects2 = ax.bar(x+width/2, means[:9], yerr=stds[:9],width=width,label='SNR: {} dB'.format(10))
ax.set_ylabel('Fraction of Mean PE to noise (0 MUs) PE')
ax.set_xlabel('# Motor Unts')
ax.set_title('PE variation with number of MUs - {} repeats'.format(repeats))
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.plot(np.arange(len(labels)),0.8*np.ones(len(labels)),'k--',label='0.8 of noise PE')
ax.plot(np.arange(len(labels)),0.9*np.ones(len(labels)),'k-',label='0.9 of noise PE')
ax.legend(loc='lower left')

fig.tight_layout()
plt.show()

#%% - Examine how PE varies with SNR: [0,0.2,0.4,...,2] ratio of signal amp.
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
import readers 
import Denoise as dn
import entropies as PE
import numpy as np
import matplotlib.pyplot as plt

def noise(ratio):
    ratio[np.where(ratio==0)[0]] = 1e-12
    SNR = 20*np.log10(np.divide(1,ratio))
    return SNR
N_MU = 10
SNR = [60,30,20,10,0,-10]#noise(np.linspace(0,2,11))
T = 5
fs = 2048
window = 2*fs

Xi = readers.GetSimData([2,1,1000,2000])[0]
ind = np.argmax(np.max(Xi,axis=0))

def generate_series(x,SNR,reference=None):
    mean = np.mean(x)
    stddev = (np.std(x)/(10**(SNR/20)))  
    if len(reference) != 0:
        mean = np.mean(reference)
        stddev = (np.std(reference)/(10**(SNR/20)))  
    noise = np.random.normal(mean,stddev,len(x))
    pattern = np.add(x,noise)
    return pattern;

mean_X,mean_N,std_X,std_N = [],[],[],[]
repeats = 10
for N_MU in [0,2,10]:
    for ratio in SNR:
        PE_X,PE_N = [],[]
        for repeat in np.arange(repeats):
            if N_MU != 0:
                args = [T,N_MU,ratio,fs]
                X = readers.GetSimData(args)[0].T 
                X = X[ind]
                X = dn.Gaussian(X,sigma=3,kernel_width=9).results()
                PE_X.append(PE.PE(X,3,window_size=window,fixed=True).results())
            else:
                N = generate_series(np.linspace(0,0,fs*T),ratio,reference=Xi)
                N = dn.Gaussian(N,sigma=3,kernel_width=9).results()
                PE_N.append(PE.PE(N,3,window_size=window,fixed=True).results())
        if N_MU != 0:
            mean_X.append(np.mean(PE_X))
            std_X.append(np.std(PE_X))
        else:
            mean_N.append(np.mean(PE_N))
            std_N.append(np.std(PE_N))
        print('SNR {0} done, Num MU {1} done'.format(ratio,N_MU))

labels = SNR
x = np.arange(len(labels))  # the label locations
width = 0.27  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width, mean_X[:6], yerr=std_X[:6], width=width, label='{} MUs'.format(2),color='blue')
rects2 = ax.bar(x, mean_X[6:], yerr=std_X[6:], width=width, label='{} MUs'.format(10),color='orange')
rects3 = ax.bar(x + width, mean_N, yerr=std_N, width=width, label='Noise',color='red')
ax.set_ylabel('Mean PE')
ax.set_xlabel('Signal to noise ratio')
ax.set_title('PE variation with SNR - {} repeats'.format(repeats))
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='lower left')

fig.tight_layout()
plt.show()

#%% - Exponential smoothing trial: alpha = [0.1,...,1.0]
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
import readers 
import Denoise as dn
import entropies as PE
import numpy as np
import matplotlib.pyplot as plt

N_MU = 10
SNR = 10
T = 1
fs = 2048
window = 800

args = [T,N_MU,SNR,fs]
X = readers.GetSimData(args)[0].T 
ind = np.argmax(np.max(X,axis=1))
X = X[ind]
alphas = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

def generate_series(x,SNR,reference=None):
    mean = np.mean(x)
    stddev = (np.std(x)/(10**(SNR/20)))  
    if len(reference) != 0:
        mean = np.mean(reference)
        stddev = (np.std(reference)/(10**(SNR/20)))  
    noise = np.random.normal(mean,stddev,len(x))
    pattern = np.add(x,noise)
    return pattern;
N = generate_series(np.linspace(0,0,len(X)),SNR,reference=X)

mean_X = []
std_X = []
mean_N = []
std_N = []
for alpha in alphas:
    X = dn.exp_smoothing(X,[alpha],order=1,kernel_width=5).results()
    N = dn.exp_smoothing(N,[alpha],order=1,kernel_width=5).results()
    PE_X = PE.PE(X,3,window_size=window).results()
    PE_N = PE.PE(N,3,window_size=window).results()
    mean_N.append(np.mean(PE_N))
    std_N.append(np.std(PE_N))
    mean_X.append(np.mean(PE_X))
    std_X.append(np.std(PE_X))
    
labels = alphas
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, mean_X, yerr=std_X, width=width, label='{} MUs'.format(N_MU))
rects2 = ax.bar(x + width/2, mean_N, yerr=std_N, width=width, label='Noise')
ax.set_ylabel('Mean PE')
ax.set_xlabel(r'Value of $\alpha$ parameter')
ax.set_title('PE variation with exponential smoothing')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.show()

#%% - Comparing Gaussian (sig=3,kw=9) and Exp smoothing (order=1,alpha=0.4) for SNRs.
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
import readers 
import Denoise as dn
import entropies as PE
import numpy as np
import matplotlib.pyplot as plt

## Find closest channel (maximum amplitude):
args = [2,1,1000,2000]
X = readers.GetSimData(args)[0].T 
ind = np.argmax(np.max(X,axis=1))

def noise(ratio):
    ratio[np.where(ratio==0)[0]] = 1e-12
    SNR = 20*np.log10(np.divide(1,ratio))
    return SNR
N_MU = 10
SNR = noise(np.linspace(0,2,11))
T = 5
fs = 2048
window = fs

def generate_series(x,SNR,reference=None):
    mean = np.mean(x)
    stddev = (np.std(x)/(10**(SNR/20)))  
    if len(reference) != 0:
        mean = np.mean(reference)
        stddev = (np.std(reference)/(10**(SNR/20)))  
    noise = np.random.normal(mean,stddev,len(x))
    pattern = np.add(x,noise)
    return pattern;

PE_NG,PE_NE,PE_XG,PE_XE = [],[],[],[]
for ratio in SNR:
    print('Run {0}, SNR: {1:.2g}'.format(np.where(SNR==ratio)[0][0]+1,ratio))
    args = [T,N_MU,ratio,fs]
    X = readers.GetSimData(args)[0].T 
    #ind = np.argmax(np.max(X,axis=1))
    X = X[ind]
    N = generate_series(np.linspace(0,0,len(X)),ratio,reference=X)
    N_G = dn.Gaussian(N,sigma=3,kernel_width=9).results()
    N_E = dn.exp_smoothing(N,order=1,args=[0.4],kernel_width=5).results()
    PE_NG.append(PE.PE(N_G,3,window,fixed=True).results())
    PE_NE.append(PE.PE(N_E,3,window,fixed=True).results())
    XG = dn.Gaussian(X,sigma=3,kernel_width=9).results()
    PE_XG.append(PE.PE(XG,3,window,fixed=True).results())
    XE = dn.exp_smoothing(X,order=1,args=[0.4],kernel_width=5).results()
    PE_XE.append(PE.PE(XE,3,window,fixed=True).results())

# Correct if needed.
xs = np.linspace(0,2,11)
filters = ['Noise, Guassian','{} MUs, Guassian'.format(N_MU),'Noise, Exponential','{} MUs, Exponential'.format(N_MU)]
[plt.errorbar(xs,means[i],yerr=stds[i],elinewidth=0.6,capthick=0.1,label='{} filter'.format(filters[i])) for i in range(len(means))]
plt.legend(loc='lower right')
plt.xlabel('Ratio of noise to signal amplitude')
plt.ylabel('Mean PE')
plt.title('Varitaion of PE for different SNR for Guassian and Exponential filters')
plt.xticks(xs)
plt.show()

#%% - Multiscale entropies
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
import readers 
import Denoise as dn
import entropies
import numpy as np
import matplotlib.pyplot as plt

N_MU = 10
SNR = 30
T = 5
fs = 2048
scales = np.arange(20)+1
window = 2000

args = [T,N_MU,SNR,fs]
X = readers.GetSimData(args)[0].T 
ind = np.argmax(np.max(X,axis=1))
X = X[ind]
def generate_series(x,SNR,reference=None):
    mean = np.mean(x)
    stddev = (np.std(x)/(10**(SNR/20)))  
    if len(reference) != 0:
        mean = np.mean(reference)
        stddev = (np.std(reference)/(10**(SNR/20)))  
    noise = np.random.normal(mean,stddev,len(x))
    pattern = np.add(x,noise)
    return pattern;
N = generate_series(np.linspace(0,0,len(X)),SNR,reference=X)
X = dn.Gaussian(X,sigma=3,kernel_width=9).results()
N = dn.Gaussian(N,sigma=3,kernel_width=9).results()

PE_X = entropies.MSE(X,scales=scales,compute_PE=True,return_means=True,window_size=window,normalised=True).results()
PE_N = entropies.MSE(N,scales=scales,compute_PE=True,return_means=True,window_size=window,normalised=True).results()

fig = plt.figure()
fig.add_axes((0,0,1,1))
plt.errorbar(scales,PE_X[0],yerr=PE_X[1],label='{} MUs'.format(N_MU))
plt.errorbar(scales,PE_N[0],yerr=PE_N[1],label='Noise')
plt.ylabel('Mean PE')
plt.xlabel('Scale')
plt.xticks(scales)
plt.title('Examining effect of scale on PE values')
plt.legend(loc='lower right')
fig.add_axes((0,-0.5,1,0.5))
plt.errorbar(scales,np.subtract(PE_N[0],PE_X[0]),yerr=np.sqrt(np.add(np.power(PE_N[1],2),np.power(PE_X[1],2))))
plt.xticks(scales)
plt.ylabel('Difference')
plt.xlabel('Scale')
plt.show()

''' Check:
PE_X_check = entropies.PE(X,3,False,window,disable_bar=False).results()
assert np.isclose(PE_X[0][0],np.mean(PE_X_check))'''

#%% - Sample entropy experimental
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
import readers 
import Denoise as dn
import entropies
import numpy as np
import matplotlib.pyplot as plt

N_MU = 10
SNR = 10
T = 1
fs = 2048
scales = np.arange(20)+1
window = 2000

args = [T,N_MU,SNR,fs]
X = readers.GetSimData(args)[0].T 
ind = np.argmax(np.max(X,axis=1))
def generate_series(x,SNR,reference=None):
    mean = np.mean(x)
    stddev = (np.std(x)/(10**(SNR/20)))  
    if len(reference) != 0:
        mean = np.mean(reference)
        stddev = (np.std(reference)/(10**(SNR/20)))  
    noise = np.random.normal(mean,stddev,len(x))
    pattern = np.add(x,noise)
    return pattern;

repeats = 1
r_mean_PE_X,r_mean_PE_N,r_mean_SE_X,r_mean_SE_N = [],[],[],[]
r_std_PE_X,r_std_PE_N,r_std_SE_X,r_std_SE_N = [],[],[],[]
for SNR in [0,10,30]:
    mean_PE_X,mean_PE_N,mean_SE_X,mean_SE_N = 0,0,0,0
    std_PE_X,std_PE_N,std_SE_X,std_SE_N = 0,0,0,0
    for repeat in range(repeats):
        args = [T,N_MU,SNR,fs]
        X = readers.GetSimData(args)[0].T[ind] 
        N = generate_series(np.linspace(0,0,len(X)),SNR,reference=X)
        X = dn.Gaussian(X,sigma=3,kernel_width=9).results()
        N = dn.Gaussian(N,sigma=3,kernel_width=9).results()
        
        SE_X = entropies.SampEn(X,m=2,r=0.3,distance='inf',use_own=True).results()
        SE_N = entropies.SampEn(N,m=2,r=0.3,distance='inf',use_own=True).results()
        PE_X = entropies.PE(X,bin_width=3,window_size=2000).results()
        PE_N = entropies.PE(N,bin_width=3,window_size=2000).results()
    
        mean_SE_X += (np.mean(SE_X))
        mean_SE_N += (np.mean(SE_N))
        mean_PE_X += (np.mean(PE_X))
        mean_PE_N += (np.mean(PE_N))
        
        std_SE_X += (np.std(SE_X))
        std_SE_N += (np.std(SE_N))
        std_PE_X += (np.std(PE_X))
        std_PE_N += (np.std(PE_N))
        
        print('SNR {0}, repeat {1} done'.format(SNR,repeat+1))
        
    r_mean_PE_X.append(mean_PE_X/repeats)
    r_mean_PE_N.append(mean_PE_N/repeats)
    r_mean_SE_X.append(mean_SE_X/repeats)
    r_mean_SE_N.append(mean_SE_N/repeats)
    r_std_PE_X.append(std_PE_X/repeats**3/2)
    r_std_PE_N.append(std_PE_N/repeats**3/2)
    r_std_SE_X.append(std_SE_X/repeats**3/2)
    r_std_SE_N.append(std_SE_N/repeats**3/2)

labels = [0,10,30]
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
fig = plt.figure()
fig.add_axes((0,0,1,1))
plt.title('PE for varying SNR')
rects1 = plt.bar(x - width/2, r_mean_PE_X, yerr=r_std_PE_X, width=width, label='{} MUs'.format(N_MU))
rects2 = plt.bar(x + width/2, r_mean_PE_N, yerr=r_std_PE_N, width=width, label='Noise')
plt.ylabel('Mean PE (normalised)')
plt.xticks(x,labels)
plt.xlabel(r'SNR (dB)')
plt.legend()
fig.add_axes((1.15,0,1,1))
plt.title('SE for varying SNR')
rects1 = plt.bar(x - width/2, r_mean_SE_X, yerr=r_std_SE_X, width=width, label='{} MUs'.format(N_MU))
rects2 = plt.bar(x + width/2, r_mean_SE_N, yerr=r_std_SE_N, width=width, label='Noise')
plt.ylabel('Mean Sample Entropy')
plt.xlabel(r'SNR (dB)')
plt.xticks(x,labels)
plt.legend()
plt.show()

#%% - On/Off signal
import time
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
import readers 
import Denoise as dn
import entropies
import numpy as np
import matplotlib.pyplot as plt
from Lempel_Ziv import LempelZiv as lz

args = [2,2,1000,2048]
Xi = readers.GetSimData(args)[0].T
ind = np.argmax(np.max(Xi,axis=1))


SNR = 1000
T = 8
fs = 2048

def generate_series(x,SNR,reference=None):
    mean = np.mean(x)
    stddev = (np.std(x)/(10**(SNR/20)))  
    if len(reference) != 0:
        mean = np.mean(reference)
        stddev = (np.std(reference)/(10**(SNR/20)))  
    noise = np.random.normal(mean,stddev,len(x))
    pattern = np.add(x,noise)
    return pattern;
x = np.linspace(0,0,fs*T)
noise = -30

ks = np.random.randint(0,50,size=2)
args = [T,1,SNR,fs]
X_one = readers.GetSimData(args,ks=np.expand_dims(ks[0],axis=0))[0].T
X_one = X_one[ind]
N = generate_series(x,noise,reference=X_one)
X_one[:int(len(X_one)/2)] = 0
X_one += N
    
args = [T,2,SNR,fs]
X_two = readers.GetSimData(args,ks=ks)[0].T
X_two = X_two[ind]
X_two[:int(len(X_two)/2)] = 0
#N = generate_series(x,noise,reference=N)
X_two += N
    
X = np.concatenate((X_one,X_two))
X = np.concatenate((X,N[:int(len(N)/2)]))

plt.plot(np.arange(len(X)),X)
plt.show()

PE_filt = entropies.PE(X,bin_width=3,normalised=True,window_size=2*fs).results()

''' Test processing time for 1 second data: 
win = (np.array([0.2,0.4,0.6,0.8])*2048).astype(int)
new_args = [1,2,10,2048]
X = readers.GetSimData(new_args)[0].T[ind]
X = dn.Gaussian(X,sigma=3,kernel_width=9).results()

timer = []
bin_size = [3,4,5]
repeats = 10
timer,std = [],[]
for bin in bin_size:
    for size in win:
        timer_repeat = []
        for repeat in range(repeats):
            begin = time.perf_counter()   
            PE_filt = entropies.PE(X,bin_width=bin,normalised=True,window_size=size).results()
            end = time.perf_counter()
            timer_repeat.append(end-begin)
        timer.append(np.mean(timer_repeat))
        std.append(np.std(timer_repeat))
        print('size {0:.1f}, bin {1} done'.format(size/2048,bin))
'''

fig = plt.figure()
fig.add_axes((0,0,1.5,0.6))
plt.title('On/Off signal - varying # of MUs. SNR = {} db'.format(noise))
plt.plot(np.arange(len(X)),X)
plt.ylabel('Signal (mV) - filtered')
fig.add_axes((0,-0.9,1.5,0.9))
plt.plot(np.arange(len(PE_filt)),PE_filt)
plt.ylabel('PE (normalised)')
plt.xlabel('Samples - {0:.0f} window size, {1:.0f} fs'.format(2*fs,fs))
plt.show()
#print('Time taken: {:.2f} sec'.format(end-begin))

'''
labels = win
x = np.arange(len(labels))
fig,ax = plt.subplots()
width = 0.27
rects1 = ax.bar(x-width,timer[:4],yerr=std[:4],width=width,label='Bin size = {}'.format(bin_size[0]))
rects2 = ax.bar(x,timer[4:8],yerr=std[4:8],width=width,label='Bin size = {}'.format(bin_size[1]))
rects3 = ax.bar(x+width,timer[8:],yerr=std[8:],width=width,label='Bin size = {}'.format(bin_size[2]))
ax.set_ylabel('Process time for 1 sec recording (sec)')
ax.set_xlabel('Window size')
ax.set_title('Live-time computation speed - {} repeats'.format(repeats))
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.show()'''

#%% - Real world data:
import time
begin = time.perf_counter()
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
import readers 
import Denoise as dn
import entropies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import trange

os.chdir('C:/Users/44790/Documents/Python Scripts/DataSets/EMG finger extensions/Electro-Myography-EMG-Dataset/raw_emg_data_unprocessed')

def generate_series(x,SNR,reference=None):
    mean = np.mean(x)
    stddev = (np.std(x)/(10**(SNR/20)))  
    if len(reference) != 0:
        mean = np.mean(reference)
        stddev = (np.std(reference)/(10**(SNR/20)))  
    noise = np.random.normal(mean,stddev,len(x))
    pattern = np.add(x,noise)
    return pattern;

index = pd.read_csv('index_finger_motion_raw.csv',header=None).to_numpy()
index_finger = np.mean(index,axis=1)[int(0.8*len(index)):]
x = np.linspace(0,0,len(index_finger))
rest = generate_series(x,30,reference=index_finger)
summed_AP = np.concatenate((index_finger,rest),axis=0)

grain = 1
course_grained = [np.mean(summed_AP[i*grain:(i+1)*grain]) for i in trange(len(summed_AP)//grain)]
filtered = dn.Gaussian(course_grained,sigma=3,kernel_width=9).results()

PE = entropies.PE(filtered,bin_width=3,window_size=300).results()
course_grained_PE = [np.mean(PE[i*grain:(i+1)*grain]) for i in trange(len(PE)//grain)]
filtered_PE = dn.Gaussian(course_grained_PE,sigma=3,kernel_width=9).results()

fig = plt.figure()
fig.add_axes((0,0,1.5,1))
plt.plot(np.arange(len(filtered)),filtered)
plt.xticks([])
fig.add_axes((0,-1.05,1.5,1))
plt.plot(np.arange(len(filtered_PE)),filtered_PE)
plt.show()

end = time.perf_counter()
print('Runtime: {:.2f} seconds'.format(end-begin))

