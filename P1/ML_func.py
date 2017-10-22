import numpy as np

def calculate_mse(y, tx, w):
    e = y - tx.dot(w)
    return 1/2*np.mean(e**2)

def calculate_rmse(y, tx, w):
    return np.sqrt(2*compute_mse(y, tx, w))

# This function finds the least square estimator w
def least_squares(y, tx):
    # w = (X'X)^(-1)X'y => (X'X)w = X'y
    # numpy.linalg.solve computes a linear matrix equation Ax=b
    # .T takes the transpose, .dot(x) do the dot product with the matrix x

    left_part = tx.T.dot(tx)
    right_part = tx.T.dot(y)

    w = np.linalg.solve(left_part, right_part)
    return w

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
