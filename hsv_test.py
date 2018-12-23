# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 15:05:51 2018

@author: karthik.venkat
"""
from skimage.feature import local_binary_pattern
# To calculate a normalized histogram 
from scipy.stats import itemfreq
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

cv2.namedWindow('Sample', cv2.WINDOW_AUTOSIZE)

hsv_mask = np.zeros(gray.shape, np.uint8)

# these ind values are the indices of the last detected hough circle within gray
hsv_mask[ind] = gray[ind]

_,thresh = cv2.threshold(hsv_mask,1,255,cv2.THRESH_BINARY)

contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
x,y,w,h = cv2.boundingRect(contours[0])

crop = gray[y:y+h,x:x+w]


radius = 3
# Number of points to be considered as neighbourers 
no_points = 8 * radius
# Uniform LBP is used
lbp = local_binary_pattern(crop, no_points, radius, method='uniform')

# Calculate the histogram
x = itemfreq(lbp.ravel())

 # Normalize the histogram
hist = x[:, 1]/sum(x[:, 1])
plt.plot(hist)

cv2.imshow("Sample", crop)

cv2.waitKey(0)
cv2.destroyAllWindows()
