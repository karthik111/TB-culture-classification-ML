# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 21:19:11 2018

@author: karthik.venkat
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
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

def get_x_train(x_raw_train_input, y_val):
    x_train = np.empty([0,BINS+26+BINS], int)
    y_train = np.empty([0], str)
    
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
        y_train = np.append(y_train, y_val)

    return x_train, y_train

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

print("starting RF for positive..")
# positive
x_raw_train_input = np.array(list(samples_pos.values()))
x_train = np.empty([0,BINS+26+BINS], int)
y_train = np.empty([0], str)
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
    y_train = np.append(y_train, 'P')


print("starting RF for negative..")
# negative
x_raw_train_input = np.array(list(samples_neg.values()))
#x_train = np.empty([0,BINS+26+BINS], int)

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
    y_train = np.append(y_train, 'N')



print("starting RF for condensation..")
x_raw_train_input = np.array(list(samples_conden.values()))
x_conden, y_conden = get_x_train(x_raw_train_input, 'W')
x_train = np.append(x_train, x_conden, axis=0)
y_train = np.append(y_train, y_conden)


print("starting RF for contamination..")
x_raw_train_input = np.array(list(samples_contam.values()))
x_contam, y_contam = get_x_train(x_raw_train_input, 'C')
x_train = np.append(x_train, x_contam, axis=0)
y_train = np.append(y_train, y_contam)

#clf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=0)

clf = svm.SVC(kernel='rbf', class_weight='balanced')
clf.fit(x_train, y_train)

"""
# Position validation cases
x_validation_pos = get_feature_array(np.array(list(samples_pos.values()))[SAMPLES:])

x_validation_neg = get_feature_array(np.array(list(samples_neg.values()))[SAMPLES:])

x_validation_conden = get_feature_array(np.array(list(samples_conden.values()))[SAMPLES:])

g_pos.score_samples(x_validation_pos)

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