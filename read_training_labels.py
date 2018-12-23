# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 16:46:45 2018

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

xls = pd.ExcelFile('..\\BMG\\11august\\11\\labelled_data.xlsx')

for sheet in xls.sheet_names[:-1]:
    print (sheet)
    df = pd.read_excel(xls, sheet)
#df2 = pd.read_excel(xls, 'Sheet2')
    
print (df[1]['A'])