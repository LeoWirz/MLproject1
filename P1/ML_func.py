import numpy as np

def standardize(data):
    # mean by features (columns)
    means_features = np.mean(data, axis=0) # axis=0 apply by column
    # substract the mean to each features
    S = data - means_features
    # standard deviation by features (column)
    std_features = np.std(S, axis=0)
    # divide by the std of each features
    S = S / std_features
    
    return S


def calculate_mse(y, tx, w):
    e = y - tx.dot(w)
    return 1/2*np.mean(e**2)

def calculate_rmse(y, tx, w):
    return np.sqrt(2*calculate_mse(y, tx, w))

# This function finds the least square estimator w
def least_squares(y, tx):
    # w = (X'X)^(-1)X'y => (X'X)w = X'y
    # numpy.linalg.solve computes a linear matrix equation Ax=b
    # .T takes the transpose, .dot(x) do the dot product with the matrix x

    left_part = tx.T.dot(tx)
    right_part = tx.T.dot(y)

    w = np.linalg.solve(left_part, right_part)
    return w


def ridge_regression(y, tx, lambda_):
    
    N = tx.shape[0]
    D = tx.shape[1]
    
    regul = 2*N*lambda_*np.identity(D)
    left = tx.T.dot(tx) + regul
    right = tx.T.dot(y)
    
    w = np.linalg.solve(left, right)
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

# Function used in the cross validation in order to create random permutation of indices to split the data
def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


# Cross validation function using the build_log_comb_poly function and a ridge regression to find the estimates w
def cross_validation(y, x, k_indices, k_fold, k, lambda_, degree):
    """return the loss of ridge regression."""
    
    # Create the indicies
    test_ind = k_indices[k]
    ind = [i for i in range(k_fold) if i is not k]
    train_ind = k_indices[ind].reshape(-1)
    
    # Partition the dataset
    y_te = y[test_ind]
    y_tr = y[train_ind]
    x_te = x[test_ind]
    x_tr = x[train_ind]
    
    # Create the polynomial X
    X_train = build_log_comb_poly(x_tr, degree)
    X_test = build_log_comb_poly(x_te, degree)
    
    # Find the estimates for the ridge regression
    w = ridge_regression(y_tr, X_train, lambda_)
    
    # Compute the loss of the train and test data
    loss_tr = calculate_rmse(y_tr, X_train, w)
    loss_te = calculate_rmse(y_te, X_test, w)
    return loss_tr, loss_te

# The idea is to create a random permutation vector of size N.
# Then we compute the ratio by taking N*ratio
# And then we index Y and X with our permutation vector
def split_data(x, y, ratio, seed=1):
   
    # set seed
    np.random.seed(seed)
    
    N = len(y)
    random_ind = np.random.permutation(N)
    split_ind = int(np.floor(ratio * N))
    training_ind = random_ind[:split_ind]
    testing_ind = random_ind[split_ind:]
    
    # We split X and Y unsing our new indices
    training_x = x[training_ind]
    training_y = y[training_ind]
    
    testing_x = x[testing_ind]
    testing_y = y[testing_ind]
    
    return training_x, testing_x, training_y, testing_y
