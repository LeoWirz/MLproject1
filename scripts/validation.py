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
        
def cross_validation(y, x, indices, fold, model_function, initial_w=[], max_iters=10, gamma=1, lambda_=1):
    """cross validation"""
    x_test = x[indices[fold]]
    y_test = y[indices[fold]]
    x_train = np.delete(x, indices[fold], axis=0)
    y_train = np.delete(y, indices[fold], axis=0)

    weights, training_loss = launch_model_function(y_train, x_train, model_function, initial_w, max_iters, gamma, lambda_)

    test_loss = calculate_mae(y_test, x_test, weights)

    y_prediction = np.dot(x_test, weights)
    y_prediction[np.where(y_prediction >= 0)] = 1
    y_prediction[np.where(y_prediction < 0)] = -1

    score = 1 - np.count_nonzero(y_test - y_prediction) / len(y_test)
    return test_loss, score

def k_fold_cross_validation(y, x, k, model_function, initial_w=[], max_iters=10, gamma=1, lambda_=1):
    """k fold cross validation"""
    num_row = y.shape[0]
    interval = int(num_row / k)
    np.random.seed(666)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k)]

    total_loss, total_score = 0,0
    for fold in range(k):
        loss, score = cross_validation(y,x,k_indices,fold, model_function, initial_w, max_iters, gamma, lambda_)
        total_loss += loss
        total_score += score
    total_loss = total_loss/k
    total_score = total_score/k 

    return total_loss, total_score