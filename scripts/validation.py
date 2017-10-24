import numpy as np
from implementations import *
from model_enum import *

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

    total_loss_tr, total_loss_te, total_score = 0,0,0
    for fold in range(k_fold):
        print("fold : " + str(fold))
        loss_tr, loss_te, score = cross_validation(y, x, k_indices, k_fold, fold, model_function, initial_w, max_iters, gamma, lambda_)
        total_loss_tr += loss_tr
        total_loss_te += loss_te
        total_score += score
    total_loss_tr = total_loss_tr/k_fold
    total_loss_te = total_loss_te/k_fold
    total_score = total_score/k_fold

    return total_loss_tr, total_loss_te, total_score