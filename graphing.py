# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 12:47:25 2020

@author: Russell Chiu
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft, fftfreq, ifft, rfftfreq, rfft
from scipy.optimize import *
import pandas as pd
from pandas import DataFrame
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from functools import reduce

model = np.load('plot data/model.npy')
raw = np.load('plot data/raw.npy')
recording = np.load('plot data/recording.npy')


tblModel = {'x':model[0], 'y':model[1]}
tblRaw = {'x':raw[0], 'y':raw[1]}
tblRecording = {'x':model[0], 'y':recording[1]}

DataFrame(tblModel, columns = ['x', 'y']).to_csv('model.csv', index=None, header=True)
# =============================================================================
# DataFrame(tblRaw, columns = ['x', 'y']).to_csv('raw.csv', index=None, header=True)
# DataFrame(tblRecording, columns = ['x', 'y']).to_csv('recording.csv', index=None, header=True)
# =============================================================================

# =============================================================================
# # Export image of Raw Data From EL_LOW_E
# plt.rcParams['axes.labelweight'] = 'normal'
# x, y = raw
# fig = plt.figure(num=None, figsize=(10, 6)) # Size in inches
# # =============================================================================
# # plt.rcParams.update({'font.family': 'Times New Roman'})
# # =============================================================================
# titlefont = {'fontsize':33, 'fontname': 'Times New Roman'} # 32-35
# labelfont = {'fontsize':26, 'fontname': 'Times New Roman'} # 26 only
# axnumfont = {'labelsize':20} # 16-24
# plt.title('Electronic Low E Recording', **titlefont)
# plt.plot(x, y)
# plt.grid()
# plt.xlabel("Time (s)", **labelfont)
# plt.ylabel("Amplitude", **labelfont)
# plt.tick_params(**axnumfont)
# # plt.axis([0, 1000, 0, 1]) # Fourier
# plt.axis([0, 9, -1.5, 1.5]) # Original
# # plt.savefig('imgs/pressureTime.svg', type='svg')
# =============================================================================
