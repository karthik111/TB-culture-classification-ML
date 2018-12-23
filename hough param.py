# -*- coding: utf-8 -*-
"""
Created on Fri May  4 19:58:56 2018

@author: karth
"""

circles = cv2.HoughCircles(d,cv2.HOUGH_GRADIENT,1,20, param1=50,param2=30,minRadius=150,maxRadius=400)