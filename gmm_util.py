# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 10:58:02 2018

@author: karthik.venkat
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
from skimage.feature import local_binary_pattern
# To calculate a normalized histogram 
from scipy.stats import itemfreq
from sklearn.preprocessing import normalize


BINS = 255
SAMPLES = 100
N_COMP = 15
#data = samples_pos["..\\BMG\\11august\\11\\DSC05820.JPG - ('B', 1)"]
global samples_pos
#hist1 = np.histogram(data, bins=BINS)

#hist2 = plt.hist(data, bins=BINS)

def get_gmm_model(x_raw_train_input, N_COMP = 8):
    x_train = np.empty([0,BINS+26+BINS], int)

    for x_raw_train in x_raw_train_input[0:SAMPLES]:
        # Build grayscale hist feature data
        x_train_gray_scale = x_raw_train[0] 
        #hist2 = plt.hist(x, bins=BINS)
        hist_gs = np.histogram(x_train_gray_scale, bins=BINS)
        x_reduced = hist_gs[0].reshape(1,-1)

        # Build LBP hist feature data
        lbp = x_raw_train[1]
        x = itemfreq(lbp.ravel())
        hist_lbp = x[:, 1]/sum(x[:, 1])
        x_reduced = np.append(x_reduced, hist_lbp.reshape(1, hist_lbp.shape[0]))[np.newaxis]
    
        # Build HSV hist feature data
        x_train_hsv = x_raw_train[2]
        hist_hsv = np.histogram(x_train_hsv[:,2].ravel(), bins=BINS)
        x_reduced = np.append(x_reduced, hist_hsv[0].reshape(1,-1))[np.newaxis]
    
        x_train = np.append(x_train, x_reduced, axis=0)


    clf = mixture.GaussianMixture(n_components=N_COMP, covariance_type='full', verbose=1)
    g_pos = clf.fit(x_train)
    return g_pos

def get_feature_array(x_raw_train_input):
    x_train = np.empty([0,BINS+26+BINS], int)

    for x_raw_train in x_raw_train_input[0:SAMPLES]:
        # Build grayscale hist feature data
        x_train_gray_scale = x_raw_train[0] 
        #hist2 = plt.hist(x, bins=BINS)
        hist_gs = np.histogram(x_train_gray_scale, bins=BINS)
        x_reduced = hist_gs[0].reshape(1,-1)

        # Build LBP hist feature data
        lbp = x_raw_train[1]
        x = itemfreq(lbp.ravel())
        hist_lbp = x[:, 1]/sum(x[:, 1])
        x_reduced = np.append(x_reduced, hist_lbp.reshape(1, hist_lbp.shape[0]))[np.newaxis]
    
        # Build HSV hist feature data
        x_train_hsv = x_raw_train[2]
        hist_hsv = np.histogram(x_train_hsv[:,2].ravel(), bins=BINS)
        x_reduced = np.append(x_reduced, hist_hsv[0].reshape(1,-1))[np.newaxis]
    
        x_train = np.append(x_train, x_reduced, axis=0)

    return x_train