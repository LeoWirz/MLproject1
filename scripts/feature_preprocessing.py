from proj1_helpers import *
from implementations import *
import numpy as np

def featuring(features):
    # replace -999 by nan
    temp_feat = np.copy(features)
    for x in temp_feat:
        np.place(x, x == -999, np.nan)
        


    stacked_feat = np.hstack([temp_feat] 
                             + [np.power(temp_feat,2)] 
                             + [np.absolute(temp_feat)**0.5]
                             + [np.arctan(temp_feat)]
                             + [add_pairwise_products(temp_feat)])
    inplace_map(np.nan_to_num, stacked_feat)
    return stacked_feat
    

def inplace_map(f, x):
    for i, v in enumerate(x):
        x[i] = f(v)
        
def add_pairwise_products(x):
    return np.hstack(
        (x,
        np.column_stack([x[:,i] * x[:,j] 
        for i in range(0,30) 
        for j in range(0,30) if i < j])))