from __future__ import division

import scipy.constants as const
import numpy as np
import scipy

from matplotlib.pyplot import plot
from scipy.io import wavfile
from IPython.core.display import HTML, display

from IPython.display import Audio

def wavPlayer(data, rate):
    display(Audio(data, rate=rate))
    
def loadSounds():
    [music_Fs,music_y] = wavfile.read('music.wav');
    [noise1_Fs,noise1_y] = wavfile.read('noise1.wav');
    [noise2_Fs,noise2_y] = wavfile.read('noise2.wav');

    convert_16_bit = float(2**15);
    music_y = music_y / (convert_16_bit + 1.0); 
    noise1_y = noise1_y / (convert_16_bit + 1.0); 
    noise2_y = noise2_y / (convert_16_bit + 1.0); 
    
    return music_Fs, music_y, noise1_y, noise1_Fs, noise2_y, noise2_Fs;

def recordAmbientNoise(noise_y,noise_Fs,numberOfMicrophones):

    alpha = [
            0.4129,-0.5696,-0.9515,-0.1888,
            -0.3027,   -0.2989,   -0.8328,    0.9131,
            0.7103,   -0.5586,   -0.4025,    0.2244,    
            0.4366,    0.4564,   -0.7546,    0.5334,
            -0.6721,   -0.8024,    0.7813,   -0.7830,
            -0.9236,   -0.3164,   -0.8752,   -0.3698,
            0.2823,    0.7111,    0.8452,    0.7691,
            -0.3228,    0.3952,   -0.9417,    0.6232,
            0.0810,    0.2314,   -0.3761,    0.0993,
            -0.7440,   -0.1825,    0.0488,   -0.9025,
            0.4503,   -0.9074,   -0.6182,   -0.2698,
            0.6373,    0.4597,   -0.1472,    0.0566,
            0.8363,   -0.5103];
    micSignals = np.zeros((len(noise_y),numberOfMicrophones));
    
    for i in range(numberOfMicrophones):
        if alpha[i] < 0 :
            micSignals[:,i] = discreteLPF(noise_y, abs(alpha[i]));
        else:
            micSignals[:,i] = discreteHPF(noise_y, abs(alpha[i]));
            
    return micSignals;
    
def discreteLPF(input,alpha):
    output = np.zeros(len(input));
    chi = 0.2;

    #    initial conditions
    output[0] = 0;
    for i in range(len(input)-1):
        output[i+1] = output[i] + alpha*(input[i+1] - output[i])+chi*input[i+1]*input[i];
    return output;
    
def discreteHPF(input,alpha):
    output = np.zeros(len(input));
    chi = 0.1;

    #    initial conditions
    output[0] = 0;
    for i in range(len(input)-1):
        output[i+1] = alpha*(output[i] + input[i+1] - input[i])+chi*input[i+1]*input[i];
    return output;

    