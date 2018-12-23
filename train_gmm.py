# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 21:19:11 2018

@author: karthik.venkat
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture

BINS = 255
SAMPLES = 100
N_COMP = 8
data = samples_pos["..\\BMG\\11august\\11\\DSC05820.JPG - ('B', 1)"]

hist1 = np.histogram(data, bins=BINS)

#hist2 = plt.hist(data, bins=BINS)

print("starting GMM for positive..")
# positive
x_raw_train = np.array(list(samples_pos.values()))
x_train = np.empty([0,BINS], int)
for x in x_raw_train[0:-10]:
    #hist2 = plt.hist(x, bins=BINS)
    hist2 = np.histogram(x, bins=BINS)
    x_reduced = hist2[0].reshape(1,-1)
    x_train = np.append(x_train, x_reduced, axis=0)

clf = mixture.GaussianMixture(n_components=N_COMP, covariance_type='full', verbose=1)
g_pos = clf.fit(x_train)
x_pos = x_reduced.copy()
g_pos.score(x_reduced)


# negative
print("starting GMM for negative..")
x_raw_train = np.array(list(samples_neg.values()))
x_train = np.empty([0,BINS], int)
for x in x_raw_train[0:SAMPLES]:
    #hist2 = plt.hist(x, bins=BINS)
    hist2 = np.histogram(x, bins=BINS)
    x_reduced = hist2[0].reshape(1,-1)
    x_train = np.append(x_train, x_reduced, axis=0)

clf = mixture.GaussianMixture(n_components=N_COMP, covariance_type='full', verbose=1)
g_neg = clf.fit(x_train)
x_neg = x_reduced.copy()
g_neg.score(x_reduced)


# precip
print("starting GMM for precip..")
x_raw_train = np.array(list(samples_precip.values()))
x_train = np.empty([0,BINS], int)
for x in x_raw_train:
    #hist2 = plt.hist(x, bins=BINS)
    hist2 = np.histogram(x, bins=BINS)
    x_reduced = hist2[0].reshape(1,-1)
    x_train = np.append(x_train, x_reduced, axis=0)

clf = mixture.GaussianMixture(n_components=N_COMP, covariance_type='full', verbose=1)
g_precip = clf.fit(x_train)
x_precip = x_reduced.copy()
g_precip.score(x_reduced)

# contaminated
print("starting GMM for contam..")
x_raw_train = np.array(list(samples_contam.values()))
x_train = np.empty([0,BINS], int)
for x in x_raw_train:
    #hist2 = plt.hist(x, bins=BINS)
    hist2 = np.histogram(x, bins=BINS)
    x_reduced = hist2[0].reshape(1,-1)
    x_train = np.append(x_train, x_reduced, axis=0)

clf = mixture.GaussianMixture(n_components=N_COMP, covariance_type='full', verbose=1)
g_contam = clf.fit(x_train)
x_contam = x_reduced.copy()
g_contam.score(x_reduced)

print("starting GMM for others..")
# other categories
x_raw_train = np.array(list(samples_other.values()))
x_train = np.empty([0,BINS], int)
for x in x_raw_train:
    #hist2 = plt.hist(x, bins=BINS)
    hist2 = np.histogram(x, bins=BINS)
    x_reduced = hist2[0].reshape(1,-1)
    x_train = np.append(x_train, x_reduced, axis=0)

clf = mixture.GaussianMixture(n_components=N_COMP, covariance_type='full', verbose=1)
g_other = clf.fit(x_train)
x_other = x_reduced.copy()
g_other.score(x_reduced)

# find optimum value of k based on AIC and BIC
n_components = np.arange(1, 50)
models = [mixture.GaussianMixture(n, covariance_type='full', random_state=0).fit(x_train)
          for n in n_components]
plt.plot(n_components, [m.bic(x_train) for m in models], label='BIC')
plt.plot(n_components, [m.aic(x_train) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components');

# apply PCA

from sklearn.decomposition import PCA
pca = PCA(0.99, whiten=True)
data = pca.fit_transform(x_train)
data.shape
"""
# creating positive test cases
x_raw_train = np.array(list(samples_pos.values()))
x_test = np.empty([0,BINS], int)
for x in x_raw_train[62:]:
    #hist2 = plt.hist(x, bins=BINS)
    hist2 = np.histogram(x, bins=BINS)
    x_reduced = hist2[0].reshape(1,-1)
    x_test = np.append(x_test, x_reduced, axis=0)
    
# inferencing on positive test cases
p = g_pos.score_samples(x_test)
n = g_neg.score_samples(x_test)
for i in range(0, p.shape[0]):
    print(p[i] > n[i])
    
# creating negative test cases
x_raw_train = np.array(list(samples_neg.values()))
x_test_neg = np.empty([0,BINS], int)
for x in x_raw_train[62:]:
    #hist2 = plt.hist(x, bins=BINS)
    hist2 = np.histogram(x, bins=BINS)
    x_reduced = hist2[0].reshape(1,-1)
    x_test_neg = np.append(x_test_neg, x_reduced, axis=0)
    
# inferencing on negative test cases
p = g_pos.score_samples(x_test_neg[0:100])
n = g_neg.score_samples(x_test_neg[0:100])
for i in range(0, p.shape[0]):
    print(p[i] > n[i])
    
    """