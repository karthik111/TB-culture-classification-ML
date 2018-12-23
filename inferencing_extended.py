# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 06:03:01 2018

@author: karthik.venkat
"""
#import labeldata

import numpy as np
import cv2
import circle_funcs
import pandas as pd
import sys
import gmm_util

def keywithmaxval(d):
    """ a) create a list of the dict's keys and values; 
        b) return the key with the max value"""  
    v=list(d.values())
    k=list(d.keys())
    return k[v.index(max(v))]
 
#27
#files = ["..\\BMG\\11august\\11\\DSC05827.JPG", "..\\BMG\\11august\\11\\DSC05934.JPG", "..\\BMG\\11august\\11\\DSC05935.JPG",  "..\\BMG\\11august\\12\\DSC05936.JPG", "..\\BMG\\11august\\12\\DSC05945.JPG", "..\\BMG\\11august\\13\\DSC06066.JPG"]

EXTRACTION_RADIUS = 0.7 # 70% of total area of circle to be used for feature extraction
DISPLAY = False
files_in_sample = ["..\\BMG\\11august\\11\\DSC05827.JPG", "..\\BMG\\11august\\11\\DSC05829.JPG", "..\\BMG\\11august\\11\\DSC05835.JPG", "..\\BMG\\11august\\11\\DSC05836.JPG", "..\\BMG\\11august\\11\\DSC05840.JPG"]

files_out_sample = ["..\\BMG\\11august\\11\\DSC05867.JPG", "..\\BMG\\11august\\11\\DSC05919.JPG", "..\\BMG\\11august\\11\\DSC05924.JPG", "..\\BMG\\11august\\11\\DSC05933.JPG"] #, "..\\BMG\\11august\\11\\IMG_5869.JPG"]

files_new_set1 = ["..\\BMG\\11august\\12\\DSC05936.JPG", "..\\BMG\\11august\\12\\DSC05943.JPG", "..\\BMG\\11august\\12\\DSC05954.JPG", "..\\BMG\\11august\\12\\DSC06003.JPG", "..\\BMG\\11august\\12\\IMG_5987.JPG"]

files_new_set2 = ["..\\BMG\\11august\\13\\DSC06066.JPG", "..\\BMG\\11august\\13\\DSC06070.JPG", "..\\BMG\\11august\\13\\DSC06090.JPG", "..\\BMG\\11august\\13\\DSC06094.JPG", "..\\BMG\\11august\\13\\IMG_6097.JPG"]

files = files_out_sample
y_hat = dict()

for f in files:
    print("Image file: "+ f)
    
    
    img = cv2.imread(f)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    kernel = np.array ([[1,0,-1], [1,0,-1], [1,0,-1]])  
           
    d = cv2.filter2D(img,-1,kernel)
    dg = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY) 

    #print("before")
  
    rect_image = np.zeros(gray.shape, np.uint8)

    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,285, param1=51,param2=48,minRadius=100,maxRadius=175)
    # To Do:  Decrease min radius and max radius in step counts of 5 until exactly 48 circles are identified - no less, no More. 

    #circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,285, param1=51,param2=48,minRadius=95,maxRadius=225)
    
    if circles.shape[1] != 48:
        circles = repeat_hough(circles, gray)
        print(f + " " + str(circles.shape[1]))
        if circles.shape[1] == 48:
            #reworked_samples = np.append(reworked_samples, f)
            print("done")
        else:
            #discarded_samples = np.append(discarded_samples, f)
            continue
        
    #print("after")
    circles = np.uint16(np.around(circles))
    
    sample = []
    ind = []
    circle_image = []
    rect_mask = []
    rect_image = []

    rect_mask = np.zeros(gray.shape, np.uint8)

    xs, ys = circle_funcs.get_sorted_circles(circles)
    
        
    if (DISPLAY):
        
        #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        #cv2.imshow('image',img)
    
        cv2.namedWindow('Sample', cv2.WINDOW_NORMAL)
        
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(img,(i[0],i[1]),i[2],(0,0,255),5)
        
          # draw an extraction zone
        cv2.circle(img,(i[0],i[1]),int(i[2]* EXTRACTION_RADIUS),(255,0,00),5)
        
        if (DISPLAY):
            cv2.imshow("Sample", img)
        # draw the center of the circle
        #cv2.circle(dg,(i[0],i[1]),2,(255,255,255),-1)
        label = str(circle_funcs.get_plate_loc(xs, ys,i[0],i[1]))
        #print(label)
        key = f + " - " + label
    
        rect_image = np.zeros(gray.shape, np.uint8)
    
        # draw an white 255 circle within the black image that will be used to mask
        cv2.circle(rect_image,(i[0],i[1]), int(i[2] * EXTRACTION_RADIUS),255,thickness=-1)
    
        # apply the OR operator to get the white pixels using the mask
        circle_image = rect_image | rect_mask
    
        # get all pixel coordinates where value is 255
        ind = np.where(circle_image==255)
    
        # get the pixel values for those coordinates from the original image - full 3 col or grayscale only?
        sample = gray[ind]
    
        lbp = get_lbp(gray, ind)
        
        # change the color of the labelled circle back to white
        cv2.circle(img,(i[0],i[1]),i[2],(255,255,255),5)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[ind]
        
        # collect the features in a list
        data = (sample, lbp, hsv)
        
        x_reduced = get_feature_array(np.array(data)[np.newaxis])
        d = dict()
        
        d['P'] = g_pos.score_samples(x_reduced)
        d['N'] = g_neg.score_samples(x_reduced)
        d['W'] = g_conden.score_samples(x_reduced)
        d['C'] = g_contam.score_samples(x_reduced)
    
        label = str(keywithmaxval(d))
        y_hat[key] = label
        
        """
        if p > n:
            label = "P"
        else:
            label = "N"
        """
        
        diff=""
        #diff = d['P'] - d['N']
        #diff = round(np.log(round(diff[0],2)),2)
        
        cv2.putText(img, label + " " + str(diff) , (i[0],i[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(255,255,255),3,cv2.LINE_AA)
        
        # store in a dict object
        well_label = f + " - " + label
        #samples[well_label] = sample
        
        #cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
        
        if (DISPLAY):
            cv2.imshow("Sample", img)
            
    if (DISPLAY):
        key = cv2.waitKey(0)
    
    for c in range(1,3):
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
sys.exit(0) 