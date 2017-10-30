# -*- coding: utf-8 -*-
import numpy as np

def magic(models, iterations, lambda_, degrees):
    """
        This function creates for each of the 4 models the polynomial matrix X,
        with the cross therm, logarithm and square root.

        It then compute the weights for each models using the logistic regression
    """

    W = []
    for ind, m in enumerate(models):

        if ind is 0:
            print("------------ LABEL 0 ---------------")
            m[0] = build_comb_poly_log_sqrt_m(m[0], degrees[ind])

            # Initialize weights for the logistic regression
            w_init = np.zeros(m[0].shape[1])

            #Run logistic regression with 500 iteration
            print("running logistic regression for model label 0")
            lossF0, w  = logistic_regression(m[1], m[0], w_init, iterations, lambda_)
            W.append(w)

        if ind is 1:
            print("------------ LABEL 1 ---------------")
            m[0] = build_comb_poly_log_sqrt_m(m[0], degrees[ind])

            # Initialize weights for the logistic regression
            w_init = np.zeros(m[0].shape[1])

            #Run logistic regression with 500 iteration
            print("running logistic regression for model label 1")
            lossF1, w  = logistic_regression(m[1], m[0], w_init, iterations, lambda_)
            W.append(w)

        if ind is 2:
            print("------------ LABEL 2 ---------------")
            m[0] = build_comb_poly_log_sqrt_m(m[0], degrees[ind])

            # Initialize weights for the logistic regression
            w_init = np.zeros(m[0].shape[1])

            #Run logistic regression with 500 iteration
            print("running logistic regression for model label 2")
            lossF2, w  = logistic_regression(m[1], m[0], w_init, iterations, lambda_)
            W.append(w)

        if ind is 3:
            print("------------ LABEL 3 ---------------")
            m[0] = build_comb_poly_log_sqrt_m(m[0], degrees[ind])

            # Initialize weights for the logistic regression
            w_init = np.zeros(m[0].shape[1])

            #Run logistic regression with 500 iteration
            print("running logistic regression for model label 3")
            lossF3, w  = logistic_regression(m[1], m[0], w_init, iterations, lambda_)
            W.append(w)

    losses = [lossF0, lossF1, lossF2, lossF3]

    return W, losses

# ----------------- DATA CLEANING ------------------------

def clean(X):
    """This function replace the -999 in the first column by the median of the column.
    It also remove a column if there are only -999 or 0 values

    """

    # 1: clean the first column, put the median instead of
    median = np.median(X[ X[:,0] != -999,0])
    np.place(X[:,0], X[:,0] == -999, median)

    # 2: delete columns with only -999 values and 0 values
    D = X.shape[1]
    N = X.shape[0]
    ind = []

    for i in range(D):
        n999 = len(X[ X[:,i] == -999, i])
        n0 = len(X[ X[:,i] == 0, i])
        if (n999 == N) or (n0 == N):
            ind.append(i)

    X = np.delete(X, ind, 1)


    return X

def findIndLabel(A, label, column):
    """
    Given a matrix, this function will return a vector of indiced where the given column in
    the matrix is equal to the given label
    """
    N = A.shape[0]
    ind = []

    for i in range(N):
        if A[i,column] == label:
            ind.append(i)

    return ind

def split_by_label(X, y, ids, labels, column):
    """
    This function splits the dataset X, the observation Y and the ids with respect
    to the labels present in a given column

    It returns an array of models,
    where models[0] is the first model,
    models[0][0] is the X matrix of the first model
    models[0][1] is the Y vector of the first model
    models[0][2] is the IDS vector of the first model
    """

    m0 = [ None, None, None]
    m1 = [ None, None, None]
    m2 = [ None, None, None]
    m3 = [ None, None, None]

    models = [m0, m1, m2, m3]

    # split the models
    for (ind_lab, l) in enumerate(labels):

        ind = findIndLabel(X, l, column)

        models[ind_lab][0] = X[ind]
        models[ind_lab][1] = y[ind]
        models[ind_lab][2] = ids[ind]

    # Remove the column with the label
    for m in models:
        m[0] = np.delete(m[0], column, 1)


    return models

def standardize_m(data):
    # mean by features (columns)
    means_features = np.mean(data, axis=0) # axis=0 apply by column
    # substract the mean to each features
    S = data - means_features
    # standard deviation by features (column)
    std_features = np.std(S, axis=0)
    # divide by the std of each features
    S = S / std_features

    return S

def build_poly_m(x, degree):


    N = x.shape[0]
    # first let's create the X matrix, full of one, with the right dimensions
    X = np.ones((N,1))
    # Then we loop from 1 to d+1 (skip the first column but go up to d degree)

    for i in range(1, degree+1):
        add = x**i
        X = np.concatenate((X,add),axis=1)

    return X

# Adds the cross term x1x2 x1x3 ... x1xD ... x2x3 ... x2xD x3x4 ... ... xD-1xD
def build_comb_poly_log_sqrt_m(x, d):

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

    add1 = np.log(np.absolute(x) + 1)
    X = np.concatenate((X,add1),1)

    add2 = np.sqrt(np.absolute(x))
    X = np.concatenate((X,add1),1)

    return X


# ------------------- ML FUNCTIONS ------------------------------

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

    return loss, w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    # init parameters
    threshold = 1e-8
    losses = []

    # build tx
    tx = x
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

# Create the combinaisons of polygons (x1x1, x1x2, x1x3, x2x2, x2x3, etc)
def build_poly_combinations(x):
    indices = []

    # Create indices
    for i in range (len(x[0])):
        for t in range (i,len(x[0])):
            indices.append([t, i])
    indices = np.array(indices).T

    n = x.shape[1]
    number_of_rows = indices.shape[1] + n

    poly_x = np.zeros((len(x), number_of_rows))
    poly_x[:, :n] = x
    poly_x[:,n:n + indices.shape[1]] = x[:, indices[0]] * x[:, indices[1]]

    return poly_x


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
