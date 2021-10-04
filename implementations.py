# Useful starting lines

import numpy as np
import matplotlib.pyplot as plt
#from proj1_helpers import *



def compute_err(y,tx,w):
    return y - tx.dot(w)


def compute_loss(y, tx, w):
    err = compute_err(y,tx,w)
    return 1/2*err.T.dot(err)/len(y)


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = compute_err(y,tx,w)
    grad = -tx.T.dot(err)/len(y)
    return grad

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def compute_logistic_pred(y,tx,w):
    return sigmoid(tx.dot(w))

def compute_logistic_gradient(y,tx,w):
    "Compute the logistic gradient"
    err = y - compute_logistic_pred(y,tx,w)
    grad = -tx.T.dot(err)
    return grad

def compute_logistic_loss(y,tx,w):
    """compute the cost by negative log likelihood."""
    loss = np.log(1+np.exp(tx.dot(w))).sum() - y.dot(tx.dot(w))
    return loss
    

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    #Lopp for computing w
    for n_iter in range(max_iters):
        grad = compute_gradient(y,tx,w)
        w = w - gamma*grad

    loss = compute_loss(y,tx,w)
    
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):

    w = initial_w
    
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx):
            # compute a stochastic gradient and loss
            grad = compute_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            
    # calculate loss
    loss = compute_loss(y, tx, w)
    
    return w, loss

def least_squares(y, tx):
    w = np.linalg.solve(tx.T.dot(tx),tx.T.dot(y))
    loss = compute_loss(y,tx,w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    N = tx.shape[0]
    D = tx.shape[1]
    
    w = np.linalg.solve(tx.T.dot(tx)+lambda_*2*N*np.eye(D),tx.T.dot(y))
    loss = compute_loss(y,tx,w) + lambda_*w.T.dot(w).squeeze()
    
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    #Lopp for computing w
    for n_iter in range(max_iters):
        grad  = compute_logistic_gradient(y,tx,w)
        w = w - gamma*grad

    loss = compute_logistic_loss(y,tx,w)
    
    return w, loss
    

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    #Lopp for computing w
    for n_iter in range(max_iters):
        grad = compute_logistic_gradient(y,tx,w) + 2*lambda_*w
        w = w - gamma*grad

    loss = compute_logistic_loss(y,tx,w) + lambda_*np.squeeze(w.T.dot(w))
    
    return w, loss







######
def batch_iter(y, tx, batch_size=1, num_batches=1, shuffle=True):
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
