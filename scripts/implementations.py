# -*- coding: utf-8 -*-
import numpy as np

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        g = compute_gradient_GD(y, tx, w)
        loss = compute_loss_GD(y, tx, w)
        w = w - gamma * g

        ws.append(np.copy(w))
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return ws[-1], losses[-1]

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w
    batch_size = 1
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            g = compute_gradient_GD(minibatch_y, minibatch_tx, w)
            loss = compute_loss_GD(y, tx, w)
            w = w - gamma * g

            ws.append(np.copy(w))
            losses.append(loss)
            
        print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
        
    return ws[-1], losses[-1]

def least_squares(y, tx):
    """calculate the least squares."""
    w = np.linalg.solve((tx.T).dot(tx),(tx.T).dot(y))
    loss = calculate_mse(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    N = y.shape[0]
    txTtx = tx.T.dot(tx)
    w = np.linalg.solve((tx.T).dot(tx) + (lambda_*np.identity(tx.shape[1])),(tx.T).dot(y))
    loss = calculate_mse(y, tx, w)
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    # init parameters
    threshold = 1e-8
    losses = []

    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # compute the y estimate and map them to the probability range (0,1)
        # here we use this weird form to avoid memory/numerical issues
        y_est = tx.dot(w)
        sigmoid = np.exp(-np.logaddexp(0, -y_est))

        # compute the loss (the weird for is just to avoid memory issues)
        loss = y*np.log(sigmoid+1e-1)+(1-y)*(np.log(1-sigmoid+0.1))
        loss = -np.sum(loss)

        # compute the gradient using the sigmoid function
        grad = tx.T.dot(sigmoid-y)

        # update the weights
        w = w - gamma * grad
        
        # log info
        if iter % 10 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))

        # converge criteria
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    return w,loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iter, gamma):
    # init parameters
    threshold = 1e-8
    losses = []

    # build tx
    tx = tx
    w = initial_w

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # log info
        if iter % 500 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # converge criteria
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    # visualization
    print("The loss={l}".format(l=penalized_loss(y, tx, w)))
    return losses[-1], w

    return w, loss





######################## helpers ########################
def calculate_mse(y, tx, w):
    e = y - tx.dot(w)
    return 1/2*np.mean(e**2)

def calculate_rmse(y, tx, w):
    return np.sqrt(2*compute_mse(y, tx, w))

def calculate_mae(y, tx, w):
    e = y - tx.dot(w)
    return np.mean(np.abs(e))

def compute_loss_GD(y, tx, w):
    return calculate_mse(y, tx, w)

def compute_gradient_GD(y, tx, w):
    """Compute the gradient."""
    e = y - tx.dot(w)
    return -(1.0/tx.shape[0])*tx.T.dot(e)

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
            
def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((x.shape[0], 1))
    for i in range(1,degree + 1):
        xpoly = x**i
        poly = np.concatenate((poly,xpoly), axis=1)   
    return poly

def standardize(x, mean_x=None, std_x=None):
    """Standardize the original data set."""
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    x[:, std_x>0] = x[:, std_x>0] / std_x[std_x>0] #to avoid division by zero and NaN values 
    
    return x, mean_x, std_x


def penalized_loss(y, tx, w, lambda_):
    return calculate_loss(y, tx, w) + lambda_ * w.T.dot(w)

def penalized_logistic_regression(y, tx, w, lambda_):
    loss = penalized_loss(y, tx, w, lambda_)
    grad = calculate_gradient(y, tx, w) + lambda_ * 2 * w
    return loss, grad

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    loss, grad = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma*grad
    return loss, w