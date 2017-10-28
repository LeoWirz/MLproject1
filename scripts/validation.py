import numpy as np
from implementations import *
from model_enum import *
from feature_preprocessing import *
from plots import *

def launch_model_function(y, x, model_function, initial_w=[], max_iters=10, gamma=1, lambda_=1):
    
    if model_function == Model.LEAST_SQUARES:
        return least_squares(y,x)
    elif model_function == Model.LEAST_SQUARES_GD:
        return least_squares_GD(y,x,initial_w,max_iters,gamma)
    elif model_function == Model.LEAST_SQUARES_SGD:
        return least_squares_SGD(y,x,initial_w,max_iters,gamma)
    elif model_function == Model.LOGISTIC_REGRESSION:
        return logistic_regression(y,x,initial_w,max_iters,gamma)
    elif model_function == Model.REG_LOGISTIC_REGRESSION:
        return reg_logistic_regression(y,x,lambda_,initial_w,max_iters,gamma)
    elif model_function == Model.RIGDE_REGRESSION:
        return ridge_regression(y,x, lambda_)
    else:
        print("no function")
        return [0,0]

def cross_validation(y, x, k_indices, k_fold, k, model_function, initial_w=[], max_iters=10, gamma=1, lambda_=1):
    """return the loss of ridge regression."""

    x_te = x[k_indices[k]]
    y_te = y[k_indices[k]]
    x_tr = np.delete(x, k_indices[k], axis=0)
    y_tr = np.delete(y, k_indices[k], axis=0)
    
    # Find the estimates for chosen model
    w, training_loss = launch_model_function(y_tr, x_tr, model_function, initial_w, max_iters, gamma, lambda_)

    # Compute the loss of the train and test data
    loss_tr = calculate_rmse(y_tr, x_tr, w)
    loss_te = calculate_rmse(y_te, x_te, w)

    y_prediction = np.dot(x_te, w)
    y_prediction[np.where(y_prediction >= 0)] = 1
    y_prediction[np.where(y_prediction < 0)] = -1

    score = 1 - np.count_nonzero(y_te - y_prediction) / len(y_te)

    return loss_tr, loss_te, score

def k_fold_cross_validation(y, x, k_fold, model_function, initial_w=[], max_iters=10, gamma=1, lambda_=1):
    """k fold cross validation"""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(666)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]

    total_loss_tr, total_loss_te, total_score = [],[],[]
    for fold in range(k_fold):
        #print("fold : " + str(fold))
        loss_tr, loss_te, score = cross_validation(y, x, k_indices, k_fold, fold, model_function, initial_w, max_iters, gamma, lambda_)
        total_loss_tr.append(loss_tr)
        total_loss_te.append(loss_te)
        total_score.append(score)
    #total_loss_tr = total_loss_tr/k_fold
    #total_loss_te = total_loss_te/k_fold
    #total_score = total_score/k_fold

    #return [np.mean(total_loss_tr), np.std(total_loss_tr)],[np.mean(total_loss_te), np.std(total_loss_te)],[np.mean(total_score), np.std(total_score)]
    return total_loss_tr, total_loss_te, total_score

def bias_variance_demo(y, x, d, model_function, initial_w=[], max_iters=0, gamma=1, lambda_=1, ratio = 0.005):
    
    # define parameters
    seeds = range(5)
    num_data = 10000
    ratio_train = ratio
    degrees = range(1, d+1)
    
    # define list to store the variable
    rmse_tr = np.empty((len(seeds), len(degrees)))
    rmse_te = np.empty((len(seeds), len(degrees)))
    
    for index_seed, seed in enumerate(seeds):
        #print("seed number : {}".format(index_seed))
        np.random.seed(seed)
        
        x_tr, x_te, y_tr, y_te = split_data(x, y, ratio_train, seed)

        # Loop through the different degrees for the polynomial and find the error
        for index_degree, degree in enumerate(degrees):
            print("degree {}".format(degree))
            # form polynomial data
            X_tr = build_log_comb_poly(x_tr, degree)
            X_te = build_log_comb_poly(x_te, degree)
            # least square
            w, training_loss = launch_model_function(y_tr, X_tr, model_function, initial_w, max_iters, gamma, lambda_)
            # calculate the rmse for train and test
            rmse_tr[index_seed, index_degree] = calculate_rmse(y_tr, X_tr, w)
            rmse_te[index_seed, index_degree] = calculate_rmse(y_te, X_te, w)

    bias_variance_decomposition_visualization(degrees, rmse_tr, rmse_te, d, lambda_)

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