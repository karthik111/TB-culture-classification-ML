# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 13:32:25 2018

@author: karthik.venkat
"""

for i in np.arange(10):
    if i % 2 == 0:
        print(i)
    elif i % 3 == 0: 
        print(i)
    else: 
        continue
    print(" i is even")


#np.savetxt("reworked_samples.txt", reworked_samples, fmt="%s")
    
for s in samples_pos.values():
plt.hist(s, bins=25)
plt.show()

i = 1
while i<10:
    print(i)
    i+=1
    
    
for r in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
    #print (r)
    for c in range(1,9):
        print("('" + r + "', " + str(c) + ")")
        


key = '..\\BMG\\11august\\11\\DSC05817.JPG' + " - " + "('B', 4)"
for sample_dict in [samples_neg, samples_pos]:
    print("Found in: ") 
    if key in sample_dict: 
        del sample_dict[key]