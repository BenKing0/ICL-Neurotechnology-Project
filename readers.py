import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

def GetOTBMat(arguments):
    """
    Extracts the EMG and sampling frequency from the OTB output MAT file.
    """
    import numpy as np
    import scipy.io as sio
    import sys
    if len(arguments) != 2:
        raise Exception('GetOTBMat needs two arguments, path and filename')
    path = arguments[0]
    filename = arguments[1]
    if filename[-3:] != 'mat':
        raise Exception('File should end .mat, is this a matlab file?')
    if sys.platform.startswith('linux'):
        if path[-1] != '/': path += '/'
    elif sys.platform.startswith('win'):
        if path[-1] != '\\': path += '\\'  
    file = path + filename
    mat = sio.loadmat(file)
    
    #Go through channels to find the emg recording length
    #(some channels may be empty)
    emgLength = 0
    for i in range(np.shape(mat['SIG'])[0]):
        for j in range(np.shape(mat['SIG'])[1]):
            if np.size(mat['SIG'][i,j]) > emgLength:
                emgLength = np.size(mat['SIG'][i,j])
                
    #Need to change from OTB matlab cell structure to a 2D matrix
    emg = np.zeros([emgLength,np.size(mat['SIG'])])
    count = 0
    for i in range(np.shape(mat['SIG'])[0]):
        for j in range(np.shape(mat['SIG'])[1]):
            emg[:,count] = mat['SIG'][i,j]
            count += 1 
            
    return emg,int(mat['fsamp'][0,0]),0

def GetOTBMatDecomp(arguments):
    """
    Extracts the EMG and sampling frequency from the OTB output MAT file.
    Includes DEMUSE decomposition MUs
    """
    import numpy as np
    import scipy.io as sio
    import sys
    if len(arguments) != 2:
        raise Exception('GetOTBMat needs two arguments, path and filename')
    path = arguments[0]
    filename = arguments[1]
    if filename[-3:] != 'mat':
        raise Exception('File should end .mat, is this a matlab file?')
    if sys.platform.startswith('linux'):
        if path[-1] != '/': path += '/'
    elif sys.platform.startswith('win'):
        if path[-1] != '\\': path += '\\'  
    file = path + filename
    mat = sio.loadmat(file)
    sPulses = np.squeeze(mat['MUPulses'])
    MUs = []
    MUs.append(mat['IPTs'].T)
    
    #Go through channels to find the emg recording length
    #(some channels may be empty)
    emgLength = 0
    for i in range(np.shape(mat['SIG'])[0]):
        for j in range(np.shape(mat['SIG'])[1]):
            if np.size(mat['SIG'][i,j]) > emgLength:
                emgLength = np.size(mat['SIG'][i,j])
                
    #Need to change from OTB matlab cell structure to a 2D matrix
    emg = np.zeros([emgLength,np.size(mat['SIG'])])
    count = 0
    for i in range(np.shape(mat['SIG'])[0]):
        for j in range(np.shape(mat['SIG'])[1]):
            emg[:,count] = mat['SIG'][i,j]
            count += 1 
    
    fSamp = int(mat['fsamp'][0,0])
    
    file = path + r"iEMG_decomp_wire.ann"
    timeStamps = np.loadtxt(file)
    
    if path[-3:-2].isalpha():
        lib = int(path[-2])
    else:
        lib = int(path[-3:-1])

    
    file = r"C:\Users\alexk\Documents\sEMG Examples\correctionFile"
    mat = sio.loadmat(file)
    adjLag = int(mat['lagValues'][0,lib-1])
    adjSamp = int(mat['resampleValues'][0,lib-1])
    timeStamps[:,1] = timeStamps[:,1] - 1
    timeStamps[:,0] = (timeStamps[:,0] * (10000 * fSamp)/adjSamp) + adjLag
    timeStamps = timeStamps[np.squeeze(np.argwhere(timeStamps[:,0]>0)),:]
    timeStamps = timeStamps[np.squeeze(np.argwhere(timeStamps[:,0]<emg.shape[0])),:]
    iPulses = []
    for i in range(int(np.max(timeStamps[:,1]))+1):
        stamps = np.squeeze(timeStamps[np.argwhere(timeStamps[:,1]==i),0])
        iPulses.append(stamps.astype('int'))
    iEMG = np.zeros([emg.shape[0],int(np.max(timeStamps[:,1]))+1])
    for i in range(int(np.max(timeStamps[:,1]))+1):
        spikes = timeStamps[np.argwhere(timeStamps[:,1]==i),0].astype('int')
        iEMG[spikes,i] = 1
      
    
    MUs.append(iEMG)
    MUs.append(sPulses)
    MUs.append(iPulses)
        
    return emg,fSamp,MUs

def GetSimData(arguments,ks=None):
    """
    Uses the generateMUAPtrain class to generate 2048Hz sampled 192 chan HD-sEMG.
    """
    from simulation import generateMUAPtrain
    if len(arguments) != 4:
        raise Exception('GetSimData needs four arguments, sampleTime, numMU, SNR, sampFreq')
    sampleTime = arguments[0] 
    numMU = arguments[1]
    SNR = arguments[2]
    fs = arguments[3]
    gen = generateMUAPtrain(sampleTime,numMU,fs,chosen_MUs=ks)
    gen.initSim()
    X = gen.returnInput(SNR)
    Y = []
    Y.append(gen.returnOutput())
    Y.append(gen.templates)
    return X,fs,Y