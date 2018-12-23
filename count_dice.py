# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 10:34:56 2018

@author: karthik.venkat
"""

import cv2
import sys

# read command-line arguments
filename = "star_mod.jpg" 
filename = "crop.jpg" 
#filename = '08-dice.jpg'
t = 200
t = 90

# read original image
img = cv2.imread(filename)

# create binary image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
(t, binary) = cv2.threshold(blur, t, 255, cv2.THRESH_BINARY)

histogram = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.plot(histogram)

# find contours
(_, contours, _) = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
    cv2.CHAIN_APPROX_SIMPLE)

# print table of contours and sizes
print("Found %d objects." % len(contours))
for (i, c) in enumerate(contours):
    print("\tSize of contour %d: %d" % (i, len(c)))

# draw contours over original image
cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

(x,y),radius = cv2.minEnclosingCircle(contours[0])
center = (int(x),int(y))
radius = int(radius)
cv2.circle(img,center,radius,(0,255,0),2)

# display original image with contours
cv2.namedWindow("output", cv2.WINDOW_NORMAL)
cv2.imshow("output", img)
cv2.waitKey(0)
cv2.destroyAllWindows()