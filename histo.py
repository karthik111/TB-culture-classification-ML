# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 18:03:52 2018

@author: karthik.venkat
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture

# To DO: Bin widths are not uniform. to fix?
#bins=np.arange(10,270,10)

ha1 = plt.hist(samples.get("('A', 1)"), bins=25)
ha2 = plt.hist(samples.get("('A', 2)"), bins=25)

hb1 = plt.hist(samples.get("('B', 1)"), bins=25)
hb2 = plt.hist(samples.get("('B', 2)"), bins=25)

hc1 = plt.hist(samples.get("('C', 1)"), bins=25)
hc2 = plt.hist(samples.get("('C', 2)"), bins=25)

he1 = plt.hist(samples.get("('E', 1)"), bins=25)
he2 = plt.hist(samples.get("('E', 2)"), bins=25)
he3 = plt.hist(samples.get("('E', 3)"), bins=25)
he4 = plt.hist(samples.get("('E', 4)"), bins=25)

hd1 = plt.hist(samples.get("('D', 1)"), bins=25)
hd2 = plt.hist(samples.get("('D', 2)"), bins=25)
hd3 = plt.hist(samples.get("('D', 3)"), bins=25)
hd4 = plt.hist(samples.get("('D', 4)"), bins=25)
#x0 = h[0].reshape(1,-1)

# precipitation samples (noise)
hf1 = plt.hist(samples.get("('F', 1)"), bins=25)
hf2 = plt.hist(samples.get("('F', 2)"), bins=25)
hf3 = plt.hist(samples.get("('F', 3)"), bins=25)

clf = mixture.GaussianMixture(n_components=3, covariance_type='full', verbose=1)
X_train = np.vstack([ha1[0].reshape(1,-1), ha2[0].reshape(1,-1), hb1[0].reshape(1,-1), hb2[0].reshape(1,-1), hc1[0].reshape(1,-1), hc2[0].reshape(1,-1)])
Y_train = np.ones((4,1))

X_test = np.vstack([hd1[0].reshape(1,-1), hd2[0].reshape(1,-1), hd3[0].reshape(1,-1), hd4[0].reshape(1,-1), he1[0].reshape(1,-1), he2[0].reshape(1,-1), he3[0].reshape(1,-1), he4[0].reshape(1,-1)])

X_noise = np.vstack([hf1[0].reshape(1,-1), hf2[0].reshape(1,-1), hf3[0].reshape(1,-1)])

g = clf.fit(X_train, Y_train)

samples_gen = clf.sample(6)

plt.plot(samples_gen[0][2])

plt.plot(clf.means_.T)

xt = np.arange(0,25)
clf_neg = mixture.GaussianMixture(n_components=3, covariance_type='full', verbose=1)
g_neg = clf_neg.fit(X_test)
samples_gen_neg = clf_neg.sample(6)
plt.figure(figsize=(10,5))
plt.xticks(xt)
plt.plot(clf_neg.means_.T)

clf_noise = mixture.GaussianMixture(n_components=3, covariance_type='full', verbose=1)
g_noise = clf_noise.fit(X_test)
samples_gen_noise = clf_noise.sample(6)
plt.figure(figsize=(10,5))
plt.xticks(xt)
plt.plot(clf_noise.means_.T)