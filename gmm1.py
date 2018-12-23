# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 20:57:55 2018

@author: karthik.venkat
"""

X = np.array([samples.get("('A', 1)")[:,0], samples.get("('A', 2)")[:,0],samples.get("('A', 3)")[:,0], samples.get("('A', 4)")[:,0]])

ValueError: Expected 2D array, got 1D array instead:
array=[123. 119. 120. ... 188. 188. 215.].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.

x0 = X[0].reshape(1,-1)

x1 = X[0].reshape(1,-1)

x1.shape
Out[93]: (1, 69729)

x0.shape
Out[94]: (1, 69729)

x1 = X[1].reshape(1,-1)

x1.shape
Out[96]: (1, 66957)

x1 = X[0].reshape(1,-1)

x1.shape
Out[98]: (1, 69729)

x1.shape
Out[99]: (1, 69729)

np.vstack( (x0,x1))
Out[100]: 
array([[123, 119, 120, ..., 188, 188, 215],
       [123, 119, 120, ..., 188, 188, 215]], dtype=uint8)

clf.fit(np.vstack( (x0,x1)))
Traceback (most recent call last):

ValueError: all the input array dimensions except for the concatenation axis must match exactly    