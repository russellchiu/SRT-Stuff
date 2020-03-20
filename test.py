# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:21:49 2020

@author: Administrator
"""

import numpy as np
from numpy import dot
from numpy.random import shuffle, permutation
from numpy.linalg import inv, pinv

# =============================================================================
# a = np.array([[1,2,3], [4,5,6], [7,8,9]])
# ainv = inv(a)
# x = np.array([[1,2,3]]).T
# y = dot(a, x)
# 
# =============================================================================

# =============================================================================
# a = np.array([[1,2], [4,5]])
# ainv = pinv(a)
# x = np.array([[1,2]]).T
# y = dot(a, x)
# =============================================================================

data = np.load('data/analyzedWeek20/dataTable.npy') # file num, (freq, amp, wid)

amplitudes = data[:,1]
bandwidths = data[:,2]

vector = np.concatenate((amplitudes, bandwidths), axis=1)

elec = vector[:11,:11].T
acous = vector[15:26,:11].T
trans = acous.dot(inv(elec))

# First Note Test
firstA = acous[:,0]
firstE = elec[:,0]

# print("Acoustic: ", firstA)
# print("Prediction: ", trans.dot(firstE))


# Testing any vector
# for x in range(15):
# 	twelfthA = vector[x+15,:11]
# 	twelfthE = vector[x,:11].reshape((11,1))
# 	error += (twelfthA - trans.dot(twelfthE).reshape((11,))) ** 2
# 	print("Error: ", sum(error))


ampData = vector[:,:11]
widthData = vector[:,12:-1]

# Randomizing Vectors
# =============================================================================
# for i in range(20):
# 	order = permutation(np.arange(15))
# 	
# 	el = ampData[order[:11]].T
# 	el_test = ampData[order[11:15]].T
# 	ac = ampData[order[:11] + 15].T
# 	ac_test = ampData[order[11:15] + 15].T
# 
# 	trans = ac.dot(inv(el))
# 	pred = trans.dot(el_test)
# 	err = (ac_test - pred) ** 2
# 	total_err = np.sum(err)
# 
# 	print(total_err, order)
# =============================================================================
    
# Randomizing Vectors Widths
# =============================================================================
# wElec = widthData[:15].T
# wAcous = widthData[15:].T
# =============================================================================

    
    

# =============================================================================
# # Testing New Vector
# ampSeq = np.array([ 0, 11,  9,  8, 13,  2, 10,  5,  4,  1,  6, 14,  3,  7, 12 ])
# x = ampData[ampSeq[:11]].T
# x_test = ampData[ampSeq[11:15]].T
# y = ampData[ampSeq[:11] + 15].T
# y_test = ampData[ampSeq[11:15] + 15].T
# a = y.dot(inv(x))
# print(a.dot(x) - y)
# =============================================================================
# =============================================================================
# print(np.sum((a.dot(x_test).T - y_test.T)**2))
# =============================================================================

def testOrder(order, width=False, err=False):
    vectors = ampData
    if(width):
        vectors = widthData
    
    el = vectors[order[:11]].T
    el_test = vectors[order[11:15]].T
    ac = vectors[order[:11] + 15].T
    ac_test = vectors[order[11:15] + 15].T

    trans = ac.dot(inv(el))
    pred = trans.dot(el_test)
    if(err):
        err = (ac_test - pred) ** 2
        total_err = np.sum(err)
        print(total_err, order)
    else:
        print(pred)


for i in range(20):
    order = permutation(np.arange(15))
    testOrder(order, width=True)
























