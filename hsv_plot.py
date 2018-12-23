# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 12:40:45 2018

@author: karthik.venkat
"""
import matplotlib.pyplot as plt

BINS = 255
brightHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

COL = 2

s = brightHSV[ind]

plt.figure(figsize=(20,10))

hist = np.histogram(s[:,COL].ravel(), bins=BINS)
#hist = hist[0]/sum(hist[0])
#plt.plot(hist)

#plt.hist(s[:,0], bins=255)
#plt.hist(s[:,1], bins=255)
h = plt.hist(s[:,COL], bins=100)