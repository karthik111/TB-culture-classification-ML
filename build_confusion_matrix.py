# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 12:59:01 2018

@author: karthik.venkat
"""
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.metrics import confusion_matrix
"""
def get_y_true(y_hat):
    y_hat_key = y_hat.keys()
    y_true = np.array([])
    for key in y_hat_key:
        if key in samples_pos.keys():
            true_val = 'P'
        elif key in samples_neg.keys():
            true_val = 'N'
        elif key in samples_conden.keys():
            true_val = 'W'
        elif key in samples_contam.keys():
            true_val = 'C'
        y_true = np.append(y_true, true_val)
        if (y_hat[key] == 'P' and true_val == 'N'):
            print(str(key))
            
    return y_true
"""

def get_y_true_from_xls(y_hat, f):
    y_hat_key = y_hat.keys()
    y_true = np.array([])
    
    image_labels = dict()    

    files = glob.glob("..\\**\\" + f, recursive = True)
    for file in files:
        xls = pd.ExcelFile(file)
        dir = file[0: -1*len(f)]
        for sheet in xls.sheet_names[:-1]:
            full_name = dir + sheet.strip()
            image_labels[full_name] = pd.read_excel(xls, sheet)  
    count=0
    for key in y_hat_key:
        image_path = key[0:key.index("-")-1]
        df = image_labels[image_path]
        row = key[key.index(",")-2:key.index(",")-1]
        col = key[key.index(",")+2:key.index(",")+3]
        true_val = df[int(col)][row]
        y_true = np.append(y_true, true_val)
        if (y_hat[key] == 'W' and true_val == 'P'):
            count+=1
            print(key)
    
    print("Count: %d" % (count))    
    return y_true
        
#y_true = get_y_true(y_hat)
#y_true = get_y_true_from_xls(y_hat, 'labelled_data.xlsx')
y_true = get_y_true_from_xls(y_hat, 'labelled_data - validation.xlsx')

labels=['Positive', 'Negative', 'Water', 'Contamination']
mat = confusion_matrix(y_true, np.array(list(y_hat.values())),labels=['P', 'N', 'W', 'C'])

sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted label')
plt.ylabel('True label');