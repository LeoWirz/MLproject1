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
    """ridge regression"""
    N = tx.shape[0]
    D = tx.shape[1]
    
    regul = 2*N*lambda_*np.identity(D)
    left = tx.T.dot(tx) + regul
    right = tx.T.dot(y)
    
    w = np.linalg.solve(left, right)
    loss = calculate_mse(y, tx, w)

    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma, newton=False):
    """
        Logistic regression, uses gradient descent, can use Newton's method (using hessien computation)
        in order to converge in fewer step, but much slower.
        Return the loss and updated w.
    """
    OBJECTIVE_DECREASE_THRESHOLD = 1e-6
    losses = []
    
    w = initial_w
    for i in range(max_iters):
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
    
        if i%100 == 0:
            print("Current iteration = {i}, loss before it = {l}".format(i=i, l=loss))

        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < OBJECTIVE_DECREASE_THRESHOLD:
            break
    return w, losses[-1]

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
    print("The loss={l}".format(l=penalized_loss(y, tx, w, lambda_)))
    return w, losses[-1]


######################## helpers ########################
def calculate_mse(y, tx, w):
    e = y - tx.dot(w)
    return 1/2*np.mean(e**2)

def calculate_rmse(y, tx, w):
    return np.sqrt(2*calculate_mse(y, tx, w))

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

def penalized_loss(y, tx, w, lambda_):
    return calculate_loss(y, tx, w) + lambda_ * w.T.dot(w)

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient."""
    num_samples = y.shape[0]
    loss = calculate_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    gradient = calculate_gradient(y, tx, w) + 2 * lambda_ * w
    return loss, gradient

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    w -= gamma * gradient
    return loss, w

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    w -= gamma * grad
    return loss, w

def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    pred = sigmoid(tx.dot(w)).reshape((-1,1))
    print(pred.shape)
    pred = np.diag(pred.T[0])
    r = np.multiply(pred, (1-pred))
    return tx.T.dot(r).dot(tx)