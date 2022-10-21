import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks

""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    #win is an array of Hanning window values
    win = window(frameSize)
    #hopsize is intervals at wich the integral is taken
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)   
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)    
    # cols for windowing. I.E. how many columns the spectrogram has 
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    #reshape array
    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    
    #mutiply each line by the Hanning window values
    frames *= win
    lenn=frames.shape[1]
    rez=np.fft.rfft(frames,lenn)

    return rez,lenn

def to_CSV(array,namm,samlen,anslen,step):
    name=namm.split('.')[0]
    s = open(name+"sam.csv", "w")
    a = open(name+"ans.csv", "w")
    rowsam=array.shape[1]*samlen
    rowans=array.shape[1]*anslen
    wid=array.shape[1]
    n=rowsam
    for j in range (0,n):
        s.write('cr')
        s.write(str(j))
        s.write(';ci')
        s.write(str(j))
        if(j<n-1):
            s.write(';')
    s.write('\n')
    n=rowans
    for j in range (0,n):
        a.write('cr')
        a.write(str(j))
        a.write(';ci')
        a.write(str(j))
        if(j<n-1):
            a.write(';')
    a.write('\n')
    
    lenn=int(np.floor((array.shape[0]-samlen-anslen)/step))
    
    linne=""
    for i in range(0,lenn):
        linne=""
        for j in range(0,samlen):
            #print(i*step+j)
            for k in range (0,wid):
                linne=linne+";"+str(int(np.real(array[i*step+j,k])))
                linne=linne+";"+str(int(np.imag(array[i*step+j,k])))
            if j==0 :
                linne=linne[1:]
            s.write(linne)
            linne=""
        s.write('\n')
        print(i/lenn)
        linne=""
        for j in range(0,anslen):
            for k in range (0,wid):
                linne=linne+";"+str(int(np.real(array[i*step+j+samlen,k])))
                linne=linne+";"+str(int(np.imag(array[i*step+j+samlen,k])))
            if j==0 :
                linne=linne[1:]
            a.write(linne)
            linne=""
        a.write('\n')
    s.close()
    a.close()

audiopath='eouxlswav.wav'
binsize=2**8

samplerate, samples = wav.read(audiopath)
s,lens = stft(samples, binsize)
to_CSV(s,audiopath,50,10,50)
