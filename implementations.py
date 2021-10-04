# Useful starting lines

import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *



def compute_err(y,tx,w):
    return y - tx.dot(w)


def compute_loss(y, tx, w):
    err = compute_err(y,tx,w)
    return 1/2*np.mean(err**2)


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = compute_err(y,tx,w)
    grad = -tx.T.dot(err)/len(y)
    return grad, err

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    #Lopp for computing w
    for n_iter in range(max_iters):
        grad , _ = compute_gradient(y,tx,w)
        w = w - gamma*grad

    loss = compute_loss(y,tx,w)
    
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):

    w = initial_w
    
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size, 1):
            # compute a stochastic gradient and loss
            grad, _ = compute_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            
    # calculate loss
    loss = compute_loss(y, tx, w)
    
    return w, loss

def least_squares(y, tx):
    w = np.linalg.solve(tx.T.dot(T),tx.T.dot(y))
    loss = compute_loss(y,tx,w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    return 0

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    return 0

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    return 0