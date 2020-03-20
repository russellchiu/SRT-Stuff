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
se = lambda ma, mb: np.sum((ma - mb) ** 2)

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
# 	total_err = se(ac_test, pred)
# 
# 	print(total_err, order)
# =============================================================================
    
# Randomizing Vectors Widths
# =============================================================================
# wElec = widthData[:15].T
# wAcous = widthData[15:].T
# =============================================================================

    
    

# Testing New Vector
ampSeq = np.array([ 0, 11,  9,  8, 13,  2, 10,  5,  4,  1,  6, 14,  3,  7, 12 ])
x = ampData[ampSeq[:11]].T
x_test = ampData[ampSeq[11:15]].T
y = ampData[ampSeq[:11] + 15].T
y_test = ampData[ampSeq[11:15] + 15].T
a = y.dot(inv(x))
print(se(a.dot(x_test), y_test))
print("A =", a)
# =============================================================================
# print(np.sum((a.dot(x_test).T - y_test.T)**2))
# =============================================================================

# =============================================================================
# def testOrder(order, width=False, err=False):
#     vectors = ampData
#     if(width):
#         vectors = widthData
#     el = vectors[order[:11]].T
#     el_test = vectors[order[11:15]].T
#     ac = vectors[order[:11] + 15].T
#     ac_test = vectors[order[11:15] + 15].T
#     
#     trans = ac.dot(inv(el))
#     pred = trans.dot(el_test)
#     if(err):
#         err = (ac_test - pred) ** 2
#         total_err = np.sum(err)
#         print(total_err, order)
#     else:
#         print(pred, '\n', ac_test)
# 
# 
# for i in range(20):
#     order = permutation(np.arange(15))
#     testOrder(order, width=True, err=True)
# 
# # Get Amplitude Transistion Matrix
# order = np.array([ 0, 11,  9,  8, 13,  2, 10,  5,  4,  1,  6, 14,  3,  7, 12 ])
# testOrder(order, err=True)
# =============================================================================

A = np.array([[ 2.91321689e-01,  2.41647004e+00, -6.59757215e+00,
        -5.43254673e+00,  8.42093676e-02,  4.28068780e+01,
        -2.26565771e+01,  1.70742001e+01, -1.25396802e+02,
         1.05600275e+02,  2.09129134e+01],
       [ 7.10387530e-01,  1.29290866e-01, -1.69777585e-01,
        -6.90724941e+00,  3.29133833e+01, -1.47191721e+01,
        -1.57127562e+01, -4.38815863e+01, -4.71983060e+01,
        -6.72750348e+00,  3.90808667e+01],
       [ 4.19840284e-01, -4.90282084e-01,  2.76158285e+00,
        -4.45630329e+00,  5.25472476e+00, -5.02588903e+00,
        -1.76482076e+01, -1.79952310e+01,  2.05203236e+01,
        -4.32873128e+01,  8.83305795e+01],
       [ 1.49596936e-02,  1.82627873e-01, -6.64329842e-01,
         5.23174173e+00, -3.96698988e+00,  9.81072607e+00,
        -3.86642048e+00, -1.28573189e-01, -2.30343646e+01,
         1.37625445e+01, -5.00763940e+00],
       [ 4.03364014e-01, -2.57346480e-01,  1.37452456e+00,
        -2.60036425e+00,  2.99683049e+00, -8.32114144e+00,
         9.37423871e+00,  3.51533055e+00,  1.02562027e+01,
        -2.91751146e+01,  7.63837134e+00],
       [ 3.23609615e-01, -5.57502834e-02,  8.98443474e-01,
        -7.48542077e+00,  8.83726459e+00, -7.82963978e+00,
        -7.32028224e+00, -5.41819553e+00,  5.81199290e+00,
        -2.12345431e+01,  6.40924655e+01],
       [ 2.23083529e-01, -2.25714940e-01,  1.08544023e+00,
        -4.70187823e+00,  3.03352809e+00, -8.44465488e+00,
         2.85835861e+00,  6.98454209e-01,  1.87117642e+01,
        -1.96218612e+01,  2.52146578e+01],
       [-3.91233364e-02,  4.35493346e-01, -1.27526252e+00,
         1.67350153e-01, -1.19786743e+00,  9.96590816e+00,
        -5.62734326e+00,  2.80856106e+00, -2.51813429e+01,
         1.83889584e+01,  5.33830442e+00],
       [ 2.15640800e-01, -1.87262827e-02, -2.12440863e-01,
         3.50272845e+00, -6.09428865e-01,  2.93747719e+00,
         2.65933339e+00, -4.93345556e-01, -1.33105125e+01,
         2.82748902e+00, -1.62760780e+01],
       [ 3.03466878e-01, -4.88856586e-01,  1.02124748e+00,
         7.21007577e+00, -1.24266972e+00, -3.84760084e+00,
         2.04428295e+00, -8.41375235e+00,  8.09298682e+00,
        -1.10080416e+01, -1.42917500e+01],
       [ 1.49038995e-01, -2.76076097e-01,  5.80609742e-01,
         1.92691374e+00,  1.95716476e+00, -3.72901174e+00,
        -6.36872433e-01, -6.86635750e+00,  4.31049108e+00,
        -6.31123872e+00, -8.62541831e-01]])


















