import numpy as np
from implementations import *

def featuring(features, fun=[True,True,True,True,True,True,True]):
    # replace -999 by nan
    temp_feat = np.copy(features)
    #temp_feat = standardize(temp_feat)[0]
        
    print("replacing -999 with median")
    for x in temp_feat:
        np.place(x, x == -999, np.nan)
        median = np.nanmedian(x, axis=0)    
        where_are_NaNs = np.isnan(x)
        x[where_are_NaNs] = median
        
    print("applying function")

    ones = np.ones(temp_feat.shape[0]).reshape((-1,1))    

    f1,f2,f3,f4,f5,f6,f7 = [],[],[],[],[],[],[]

    #f1 = [np.power(temp_feat,2)] 
    f2 = [np.absolute(temp_feat)**0.5]
    f3 = [np.arctan(temp_feat)]
    #f4 = [add_pairwise_products(temp_feat)]
    f5 = [np.log(np.absolute(temp_feat) + 1)]
    f6 = [log_with_neg(temp_feat)]
    f7 = [build_comb_poly(temp_feat,2)]


    stacked_feat = np.hstack( [ones] + [temp_feat] 
                             + f1
                             + f2
                             + f3
                             + f4
                             + f5
                             + f6
                             + f7
                             )

    #stacked_feat = (standardize(stacked_feat)[0])
    inplace_map(np.nan_to_num, stacked_feat)
    return stacked_feat

def log_with_neg(x):
    sign = np.sign(x)
    x = np.log(np.absolute(x)+1)
    x = np.multiply(x,sign)
    return x
    

def remove_zeros_col(x):
    return x[:,np.any(x != 0, axis=0)]

def inplace_map(f, x):
    for i, v in enumerate(x):
        x[i] = f(v)

def standardize(x, mean_x=None, std_x=None):
    """Standardize the original data set."""
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    x[:, std_x>0] = x[:, std_x>0] / std_x[std_x>0] #to avoid division by zero and NaN values 
    
    return x, mean_x, std_x
        
def add_pairwise_products(x):
    return np.hstack(
        (x,
        np.column_stack([x[:,i] * x[:,j] 
        for i in range(0,x.shape[1]) 
        for j in range(0,x.shape[1]) if i < j])))

# This function builds a polynomial version of the experience matrix X
# up to the degree d
# X = [x1 x2 x3 ... xD x1^2 x2^2 ... xD^2 x1^3 ... xD^3 ... xD^d]

def build_poly(x, degree):
    N = x.shape[0]
    # first let's create the X matrix, full of one, with the right dimensions
    X = np.ones((N,1))

    # Then we loop from 1 to d+1 (skip the first column but go up to d degree)
    for i in range(1, degree+1):
        add = x**i
        X = np.concatenate((X,add),axis=1)
    return X

# Adds the cross term x1x2 x1x3 ... x1xD ... x2x3 ... x2xD x3x4 ... ... xD-1xD
def build_comb_poly(x, d):

    D = x.shape[1]
    # first we create the matrix with the polynomial
    X = build_poly(x, d)

    # Then we add to this matric the cross term of the original matrix
    # of experience x
    for i in range(D):
        for j in range(i,D):
            if i is j:
                pass
            else:
                add = x[:,i] * x[:,j]
                d_ = X.shape[1]
                X = np.insert(X, d_, add,1)

    return X

# This adds also log(|x|+1)
def build_log_comb_poly(x,d):

    X = build_comb_poly(x,d)
    add = np.log(np.absolute(x) + 1)
    return np.concatenate((X,add),1)

def build_log_sqrt_comb_poly(x,d):
    X = build_comb_poly(x,d)

    add1 = np.log(np.absolute(x) + 1)
    X = np.concatenate((X,add1),1)

    add2 = np.sqrt(np.absolute(x))
    X = np.concatenate((X,add1),1)

    return X

# This adds also log(|x|+1)
def build_log_poly(x,d):

    X = build_poly(x,d)
    add = np.log(np.absolute(x) + 1)
    return np.concatenate((X,add),1)

# gets row with specific value (0,1,2,3) on row 22
def split_22col(y,x,ids,value):
    xv = x[np.where(x[:,22] == value)]
    yv = y[np.where(x[:,22] == value)]
    idsv = ids[np.where(x[:,22] == value)]

    return yv, xv, idsv