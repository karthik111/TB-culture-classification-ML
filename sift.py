#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 13:09:48 2017

@author: karthikeyanvenkataraman
"""
import cv2
import numpy as np

def getkeyval(key):

    if (chr(key)=='q' or chr(key)=='w'):
        for c in range(1,3):
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        if (chr(key)=='q'):
            sys.exit(1)

    return chr(key)


img = cv2.imread('../BMG/11august/12/DSC0' + str(6001) + '.jpg')

gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
d = gray

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow("image", img)

cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
cv2.imshow("gray", gray)

"""
# Sift transform
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)

img=cv2.drawKeypoints(gray,kp,d)
"""

# 2D Filters
kernel = np.ones((3,3),np.float32)/25

kernel = np.array ([[-1,0,1], [-1,0,1], [-1,0,1]])                
#kernel = np.array ([[0,1,0], [-1,0,1], [0,-1,0]])       
#kernel = np.array ([[-1,-1,-1], [0,0,0], [1,1,1]])   

#img = np.array( [ [0,0,0,255,255], [0,0,0,255,255], [0,0,0,255,255], [0,0,0,255,255],[0,0,0,255,255]])
kernel = np.array ([[1,0,-1], [1,0,-1], [1,0,-1]])  
           
d = cv2.filter2D(img,-1,kernel)
dg = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY) 

print("before")
#circles = cv2.HoughCircles(dg,cv2.HOUGH_GRADIENT,1,300,
#                            param1=80,param2=60,minRadius=120,maxRadius=270)

circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,285,
                            param1=51,param2=48,minRadius=100,maxRadius=175)
print("after")
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(dg,(i[0],i[1]),i[2],(255,255,255),2)
    # draw the center of the circle
    cv2.circle(dg,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('image',img)

cv2.namedWindow('filter', cv2.WINDOW_NORMAL)
cv2.imshow("filter", dg)

cv2.waitKey(0)
cv2.destroyAllWindows()

#cv2.imshow("Coin Detection", coins)
#key = cv2.waitKey()
#print "Coin: ", getkeyval(key)
