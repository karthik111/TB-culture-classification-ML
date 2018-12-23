# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 04:30:09 2018

@author: karthik.venkat
"""

import os
import glob
import numpy as np
import cv2
import circle_funcs
import pandas as pd
import sys
import pickle

READ_FILE_LIST = True
DISPLAY = True

# keys representing results
POSITIVE = 'p'
NEGATIVE = 'n'
CONTAM = 'c'
PRECIP = 'w'
OTHER = 'o'
QUIT = 'q'
GET_NEXT = 'z'

def repeat_hough(circles, gray):
    minRadius=100
    maxRadius=175
    n = 0
    while circles.shape[1] != 48:
        if circles.shape[1] < 48:
            if n % 2 == 0:
                minRadius -= 3
            else:
                maxRadius += 3
        else:
            if n % 2 == 0:
                minRadius += 3
            else:
                maxRadius -= 3
            
        circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,285, param1=51,param2=48,minRadius=minRadius,maxRadius=maxRadius)
        n += 1
        
        if n > 20 or circles.shape[1] == 48:
            return circles
        
def getkeyval(key):
    return chr(key)      

def load_files():
    
    samples_pos=dict(); samples_neg=dict(); samples_contam=dict(); samples_precip=dict(); samples_other=dict()
    
    if os.path.exists("samples.csv"):
        samples = pd.read_csv("samples.csv", index_col=0)
    else:
        samples = pd.DataFrame(index = files, columns=['TRAINED', 'POSITIVE', 'NEGATIVE', 'CONTAM', 'PRECIP', 'OTHER'], data=0)
        # do nothing
    
    if os.path.exists("samples_pos.pickle"):
        pickle_in = open("samples_pos.pickle","rb")
        samples_pos = pickle.load(pickle_in)
        
    if os.path.exists("samples_neg.pickle"):
        pickle_in = open("samples_neg.pickle","rb")
        samples_neg = pickle.load(pickle_in)    
      
    if os.path.exists("samples_precip.pickle"):
        pickle_in = open("samples_precip.pickle","rb")
        samples_precip = pickle.load(pickle_in)  
   
    if os.path.exists("samples_contam.pickle"):
        pickle_in = open("samples_contam.pickle","rb")
        samples_contam = pickle.load(pickle_in)  
        
    if os.path.exists("samples_other.pickle"):
        pickle_in = open("samples_other.pickle","rb")
        samples_other = pickle.load(pickle_in)     
        
    return samples, samples_pos, samples_neg, samples_contam, samples_precip, samples_other


def save_files(sample, samples_pos, samples_neg, samples_contam, samples_precip, samples_other):
    
    samples.to_csv("samples.csv")
    
    pickle_out = open("samples_pos.pickle", "wb")
    pickle.dump(samples_pos, pickle_out)
    
    pickle_out = open("samples_neg.pickle", "wb")
    pickle.dump(samples_neg, pickle_out)
    
    pickle_out = open("samples_precip.pickle", "wb")
    pickle.dump(samples_precip, pickle_out)
    
    pickle_out = open("samples_contam.pickle", "wb")
    pickle.dump(samples_contam, pickle_out)
    
    pickle_out = open("samples_other.pickle", "wb")
    pickle.dump(samples_other, pickle_out)
    
    
def remove_partial_data(f):
    
    for r in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
        for c in range(1,9):       
            key = f + " - " + "('" + r + "', " + str(c) + ")"
            for sample_dict in [samples_neg, samples_pos, samples_contam, samples_precip, samples_other]:
                if key in sample_dict: 
                    del sample_dict[key]
    
if (READ_FILE_LIST):
    files = glob.glob("..\\**\\*.JPG", recursive = True)
    np.save("sample_list", files)
else:
    files = np.load("sample_list.npy")

samples, samples_pos, samples_neg, samples_contam, samples_precip, samples_other = load_files()

reworked_samples = np.load("reworked_samples.npy")  
discarded_samples = np.load("discarded_samples.npy")
#samples_pos=dict(); samples_neg=dict(); samples_contam=dict(); samples_precip=dict(); samples_other=dict()

#samples = pd.DataFrame(index = files, columns=['TRAINED', 'POSITIVE', 'NEGATIVE', 'CONTAM', 'PRECIP', 'OTHER'], data=0)

for f in files:
    print("Image file: "+ f)
    
    # Ignore images that have been processed already
    if samples.index.contains(f) and samples.loc[f].TRAINED == 48:
        continue
    ## Ignore discarded files
    elif np.size(np.where(discarded_samples == f)[0]) != 0:
        continue
    else:
        # remove partially processed files
        remove_partial_data(f)
        samples.loc[f] = 0
      
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
            reworked_samples = np.append(reworked_samples, f)
        else:
            discarded_samples = np.append(discarded_samples, f)
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
        cv2.circle(img,(i[0],i[1]),int(i[2]*0.7),(255,0,00),5)
        
        cv2.imshow("Sample", img)
        # draw the center of the circle
        #cv2.circle(dg,(i[0],i[1]),2,(255,255,255),-1)
        label = str(circle_funcs.get_plate_loc(xs, ys,i[0],i[1]))
        #print(label)
        
        key = cv2.waitKey(0)
        print("key " + str(key))
        c = getkeyval(key)
        print("Well value: ", c)
        
        while c != POSITIVE and c!= NEGATIVE and c!=PRECIP and c!=CONTAM and c!=OTHER and c!=QUIT and c!=GET_NEXT:
            key = cv2.waitKey(0)
            print("Please press a valid key" + str(key))
            c = getkeyval(key)
            print("Key Pressed: ", c)
                        
        cv2.putText(img, label + " " + c, (i[0],i[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(255,255,255),3,cv2.LINE_AA)
    
        rect_image = np.zeros(gray.shape, np.uint8)
    
        # draw an white 255 circle within the black image that will be used to mask
        cv2.circle(rect_image,(i[0],i[1]),int(i[2]*0.7),255,thickness=-1)
    
        # apply the OR operator to get the white pixels using the mask
        circle_image = rect_image | rect_mask
    
        # get all pixel coordinates where value is 255
        ind = np.where(circle_image==255)
    
        # get the pixel values for those coordinates from the original image - full 3 col or grayscale only?
        sample = gray[ind]
        
        """ to show the cropped image, uncomment this code
        rect_mask[ind] = gray[ind]
        cv2.imwrite("mask.jpg", rect_mask)
        """
        
        # store in a dict object
        well_label = f + " - " + label
        #samples[well_label] = sample
        
        #cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
        cv2.imshow("Sample", img)
    
        
        # change the color of the labelled circle back to white
        cv2.circle(img,(i[0],i[1]),i[2],(255,255,255),5)
        
        if c == POSITIVE:
            samples.loc[f]['TRAINED'] += 1
            samples_pos[well_label] = sample
            samples.loc[f]['POSITIVE'] += 1
        elif c == NEGATIVE:
            samples.loc[f]['TRAINED'] += 1
            samples_neg[well_label] = sample
            samples.loc[f]['NEGATIVE'] += 1
        elif c == CONTAM:
            samples.loc[f]['TRAINED'] += 1
            samples_contam[well_label] = sample
            samples.loc[f]['CONTAM'] += 1
        elif c == PRECIP:
            samples.loc[f]['TRAINED'] += 1
            samples_precip[well_label] = sample
            samples.loc[f]['PRECIP'] += 1
        elif c == OTHER:
            samples.loc[f]['TRAINED'] += 1
            samples_other[well_label] = sample
            samples.loc[f]['OTHER'] += 1
        elif c == QUIT:
            print("Saving and quitting...")
            #save_files(sample, samples_pos, samples_neg, samples_contam, samples_precip, samples_other)
            for c in range(1,3):
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                sys.exit(0)
        elif c == GET_NEXT:
            print("Skpping this and going to next image...")
            break
        
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    
    
"""
To do: 
    1. Disallow invalid keys while labeling and retail focus on current well till labelled. 
    2. Resume from last saved well
"""    