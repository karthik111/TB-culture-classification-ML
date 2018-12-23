#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 13:09:48 2017

@author: karthikeyanvenkataraman
"""
import cv2
import numpy as np
import math

#import sklearn.mixture.GaussianMixture 

def getkeyval(key):

    if (chr(key)=='q' or chr(key)=='w'):
        for c in range(1,3):
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        if (chr(key)=='q'):
            sys.exit(1)

    return chr(key)

def get_sorted_circles(circles):
    xs = np.zeros(0)
    ys = np.zeros(0)
    for c in circles[0,]:
        xs = np.append(xs,c[0])
        ys = np.append(ys, c[1])
        
    return np.sort(xs), np.sort(ys)

plate_rows = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', }
def get_plate_loc(xs, ys, i, j):
    xsi = np.where(xs==i)[0][0]
    ysi = np.where(ys==j)[0][0]
    
    row = plate_rows[ math.floor(ysi/8) ]
    
    col = int(xsi/6) + 1
    return row, col
    #return i, j


#img = cv2.imread('/Users/karthikeyanvenkataraman/Desktop/memami.jpg')
#img = cv2.imread('window.jpg')

img = cv2.imread('../BMG/11august/12/DSC0' + str(6001) + '.jpg')
#img = cv2.imread('../BMG/11august/12/DSC0' + str(6002) + '.jpg')
#img =  cv2.imread('..\\BMG\\11august\\11\\IMG_5817.JPG')

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

rect_image = np.zeros(gray.shape, np.uint8)

circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,285, param1=51,param2=48,minRadius=100,maxRadius=175)
# To Do:  Decrease min radius and max radius in step counts of 5 until exactly 48 circles are identified - no less, no More. 

#circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,285, param1=51,param2=48,minRadius=95,maxRadius=225)

print("after")
circles = np.uint16(np.around(circles))
samples = dict()
sample = []
ind = []
circle_image = []
rect_mask = []
rect_image = []

rect_mask = np.zeros(gray.shape, np.uint8)

xs, ys = get_sorted_circles(circles)

for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(img,(i[0],i[1]),i[2],(255,255,255),3)
    # draw the center of the circle
    cv2.circle(img,(i[0],i[1]),2,(0,0,255),-1)
    label = str(get_plate_loc(xs, ys,i[0],i[1]))
    #print(label)
    cv2.putText(img, label, (i[0],i[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.8,(255,255,255),5,cv2.LINE_AA)
    
    rect_image = np.zeros(gray.shape, np.uint8)
    
    # draw an white 255 circle within the black image that will be used to mask
    cv2.circle(rect_image,(i[0],i[1]),i[2],255,thickness=-1)
    
    # apply the OR operator to get the white pixels using the mask
    circle_image = rect_image | rect_mask
    
    # get all pixel coordinates where value is 255
    ind = np.where(circle_image==255)
    
    # get the pixel values for those coordinates from the original image - full 3 col or grayscale only?
    sample = gray[ind]
    
    # store in a dict object
    samples[label] = sample
    
    ## TO DO: A sort based approach to get row, column indices for each circle centroid
    """
    x_l = i[0] - i[2]
    y_l = i[1] - i[2]
    x_r = x_l + 2*i[2]
    y_r = y_l + 2*i[2]
    cv2.rectangle(gray, (x_l,y_l), (x_r, y_r), (0,0,0), thickness=-1)
    """
    #img[ind] = (255,0,0)
    
cv2.imshow('image',img)
cv2.imshow("gray", gray)

cv2.namedWindow('filter', cv2.WINDOW_NORMAL)
cv2.imshow("filter", dg)

cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
cv2.imshow("mask", rect_image)

# Test the rendering of the extracted image in a black canvas and check that it renders
cv2.namedWindow('extract', cv2.WINDOW_NORMAL)
extract_image = np.zeros(img.shape, np.uint8)
cv2.circle(extract_image,(i[0],i[1]),i[2],255,thickness=-1)
extract_image[ind] = img[ind]
cv2.imshow("extract",extract_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

#gmm = 
#cv2.imshow("Coin Detection", coins)
#key = cv2.waitKey()
#print "Coin: ", getkeyval(key)
