# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 07:58:45 2018

@author: karthik.venkat
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