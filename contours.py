# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 08:28:35 2018

@author: karthik.venkat
"""

import numpy as np
import cv2 as cv
img = cv.imread('star_mod.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(img, (5, 5), 0)

ret,thresh = cv2.threshold(blur,200,255,0)
im2,contours,hierarchy = cv2.findContours(thresh, 1, 2)
cnt = contours[0]
M = cv2.moments(cnt)
print( M )

cv2.drawContours(im2, contours, -1, (0,0,255), 5)
 
cv2.imshow("Sample", im2)
cv2.waitKey(0)
cv2.destroyAllWindows()
