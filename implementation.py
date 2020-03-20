# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:43:50 2020

@author: Administrator
"""

from numpy import dot, array, arange, load
from numpy.random import shuffle, permutation
from numpy.linalg import inv, pinv
from functools import reduce
from matplotlib.pyplot import plot, axis 
from scipy.fftpack import ifft

data = load('data/analyzedWeek20/dataTable.npy') # file num, (freq, amp, wid)
A = load('./ampTransMat.npy')

prediction = lambda x: A.dot(x)

def createX(length, step):
    return arange(0, length, step)

def peak(fr, amp, wid):
    x = createX(1000, 0.05) # 0-1000 every .05
    q = 2 * 3 ** 0.5 * (2 * fr * fr + fr * wid) * (4 * fr * wid + wid * wid) ** -1
    y = lambda t: amp * (1 + q ** 2 * (t / fr - fr / t) ** 2) ** -0.5
    signal = y(x)
    return [x, signal]

def superimpose(freqs, amps, wids):
    n = freqs.size
    x = createX(1000, 0.05) # 0-1000 every .05
    y = reduce(lambda a, b: a + b, [peak(freqs[i], amps[i], wids[i])[1] for i in range(n)])
    y = y / max(y)
    return array([x, y])

frequencies = data[:,0,:-1]
amplitudes = data[:,1,:-1]
bandwidths = data[:,2,:-1]
    
lowE = superimpose(frequencies[0], prediction(amplitudes[0].reshape([11,1])), bandwidths[0])
plot(ifft(lowE[1]))
axis([-1, 5, -1, 1])

















