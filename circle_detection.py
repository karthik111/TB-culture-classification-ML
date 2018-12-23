#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 08:27:22 2017

@author: karthikeyanvenkataraman
"""

import cv2
import sys
import numpy as np
import learnt_params_load

def KNN(x_training, y_training, x_testing, K):
    """
    returns the class label for x_testing considering K nearest neighbours
    on training data (x_training, y_training)
    """
    
    d = np.sqrt(np.sum((x_training - x_testing)**2, axis=1))
    i = np.argsort(d)
    s = y_training[i[np.arange(K)],:]
    return s
    #return np.argmax(np.sum(s, axis=0))
    
x, y = 0,0
#x = x[:,np.newaxis]
#y = y[:,np.newaxis]


d = { '0': "??", '1': "10c",  '2': "20c", '3': "50c", '4': "1 R", '5': "2 R", 
     '6': "5 R Old", '7': "5 R New" 
     }


for n in np.arange(17,18):
    coins = cv2.imread('../BMG/11august/11/DSC058' + str(n) + '.jpg')

    gray_img = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
    hsv_img = cv2.cvtColor(coins, cv2.COLOR_BGR2HSV)
    
    img = cv2.medianBlur(gray_img, 5)

    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,100,
                               
    param1=100,param2=30,minRadius=250,maxRadius=500)
    
    
    if (circles is None):
        continue
    
    circles = np.uint16(np.around(circles))
    
    for i in circles[0,:]:

        old_color = np.array(coins[ i[1],i[0] ])
        hsv_vals = np.array(hsv_img[ i[1],i[0] ])
        
        # draw the outer circle

        cv2.circle(coins,(i[0],i[1]),i[2],(0,0,255),2)

        # draw the center of the circle

        cv2.circle(coins,(i[0],i[1]),2,(0,0,255),3)
    
        
        #cv2.imwrite("coins_circles.jpg", coins)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        #val = KNN(x,y,[ i[2],hsv_vals[0] ] ,1)
        #val = val[0,0]
        #c = str(val)
        #prediction = np.array_str( KNN(x,y,i[2],1) )
        #cv2.putText(coins,d[c[0]],(i[0],i[1]), font, 0.5,(255,0,0),1,cv2.LINE_AA)
        #cv2.putText(coins,c, (x+60,y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,0),1,cv2.LINE_AA)
        #cv2.namedWindow("HoughCircles", cv2.WINDOW_NORMAL)
        #cv2.imshow("HoughCircles", coins)
        
        
        #key = cv2.waitKey()
        #print "Coin: ", chr(key)
        
        #print ("Image: " + str(n)+ " ", i[0],i[1],i[2])
        #print "Color value: ", old_color 
        #print ("HSV value: ", hsv_vals)
        
        """ don't updated the trained values
        x = np.append(x, [ [i[2], hsv_vals[0]] ], axis=0)
        y = np.append(y, chr(key))
        
        #x = x[:,np.newaxis]
        y = y[:,np.newaxis]
        """
   
        cv2.circle(coins,(i[0],i[1]),i[2],(0,255,0),2)
        cv2.imwrite('test.jpg', coins)
        """
        for c in range(1,3):
            cv2.destroyAllWindows()
            cv2.waitKey(1)
          
        if (chr(key)=='q'):
            sys.exit(1)
        """  
    key = cv2.waitKey()
    cv2.destroyAllWindows()
    
        #print "Coin: ", chr(key)
#    for c in range(1,3):
#        cv2.destroyAllWindows()
#        cv2.waitKey(1)
#          
#    if (chr(key)=='q'):
#        sys.exit(1)        
# Labels - 
# <key, coin> 
# 0: <invalid coin>
# 1 5c
# 2: 10c 
# 3: 20c
# 4: 50c
# 5: 1 R
# 6: 2 R
# 7: 5 R


   