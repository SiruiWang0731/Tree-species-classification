# -*- coding: utf-8 -*-
'''This script is meant to take the first 30 bands'''

import numpy as np

#%%
# load the file
f = "5Betula_pendula_samples.npy"
ar = np.load(f, allow_pickle=True)

#%%
ar1 = ar[0, 0, 0]
ar2 = ar1.take(indices=range(0, 30), axis=2)

#%%
# slicing loop
for i in range(len(ar[:, 0, 0])):
    img = ar[i, 0, 0]
    i1 = img.take(indices=range(0, 30), axis=2)
    ar[i, 0, 0] = i1
        
#%%
# check the dimensions 
ar1 = ar[0, 0, 0]

#%%
# saving the file
np.save(f, ar)