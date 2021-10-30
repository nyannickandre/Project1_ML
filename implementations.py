# Useful starting lines

import numpy as np
from proj1_helpers import *


NULL_LAMBDA = 1e-15


def compute_err(y, tx, w):
    """Computes error"""
    return y - tx.dot(w)


def compute_loss(y, tx, w):
    """Computes loss"""
    err = compute_err(y, tx, w)
    return 1 / 2 * err.T.dot(err) / len(y)


def compute_gradient(y, tx, w):
    """Computes the gradient"""
    err = compute_err(y, tx, w)
    grad = -tx.T.dot(err) / len(y)
    return grad


def sigmoid(x):
    """Computes the sigmoid"""
    return 1.0 / (1.0 + np.exp(-x))


def compute_logistic_pred(tx, w):
    """Computes logistic prediction"""
    return sigmoid(tx.dot(w))


def compute_logistic_gradient(y, tx, w):
    """Computes logistic gradient."""
    w = w.reshape(-1)
    y = y.reshape(-1)
    err = y - compute_logistic_pred(tx, w)
    grad = -tx.T.dot(err)
    return grad


def compute_logistic_loss(y, tx, w):
    loss = np.log(1 + np.exp(tx.dot(w))).sum() - y.dot(tx.dot(w))
    return loss


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Computes the least square gradient descent
    :param y: Class labels
    :param tx: Dataset of independent variables
    :param initial_w: Initial weights
    :param max_iters: Maximum number of iterations
    :param gamma: Size of step
    :return: Weights, losses
    """
    
    masked_tx = np.nan_to_num(tx)
    # Define parameters to store w and loss
    w = initial_w
    # Lopp for computing w
    for n_iter in range(max_iters):
        grad = compute_gradient(y, masked_tx, w)
        w = w - gamma * grad

    loss = compute_loss(y, masked_tx, w)

    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Computes the least square gradient stochastic descent
    :param y: Class labels
    :param tx: Dataset of independent variables
    :param initial_w: Initial weights
    :param max_iters: Maximum number of iterations
    :param gamma: Size of step
    :return: Weights, losses
    """
    
    masked_tx = np.nan_to_num(tx)
    
    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, masked_tx):
            # compute a stochastic gradient and loss
            grad = compute_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad

    # calculate loss
    loss = compute_loss(y, masked_tx, w)

    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Computes ridge regression
    :param y: Class labels
    :param tx: Dataset of independent variables
    :param lambda_: Weight of squared weights element for computing loss
    :return: Weight, loss
    """
    N = tx.shape[0]
    D = tx.shape[1]
    tx_z = np.ma.array(tx)
    masked_tx = np.nan_to_num(tx_z)

    w = np.linalg.solve(masked_tx.T.dot(masked_tx) + lambda_ * 2 * N * np.eye(D), masked_tx.T.dot(y))
    # w = np.linalg.solve(np.dot(tx.T,tx)+lambda_*2*N*np.eye(D),np.dot(tx.T,y))
    loss = compute_loss(y, masked_tx, w) + lambda_ * w.T.dot(w).squeeze()

    return w, loss


def least_squares(y, tx):
    """
    Computes least squares. Least squares is a particular case of ridge regession with null lambda.
    :param y: Class labels
    :param tx: Dataset of independent variables
    :return: Weight, loss
    """
    w, loss = ridge_regression(y, tx, NULL_LAMBDA)

    loss = compute_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Computes logistic regression, using the gradient descent algorithm and ridge regression for the loss parameter.
    :param y: Class labels
    :param tx: Dataset of independent variables
    :param lambda_: Weight of squared weights element for computing loss
    :param initial_w: Initial weights
    :param max_iters: Maximum number of iterations
    :param gamma: Size of step
    :return: Weights, losses
    """
    
    masked_tx = np.nan_to_num(tx)
    # Define parameters to store w and loss
    w = initial_w
    # Loop for computing w
    for n_iter in range(max_iters):
        grad = compute_logistic_gradient(y, masked_tx, w) + 2 * lambda_ * w
        w = w - gamma * grad

    loss = compute_logistic_loss(y, masked_tx, w) + lambda_ * np.squeeze(w.T.dot(w))

    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Computes logistic regression, using the gradient descent algorithm
    It is a specific case of reg logistic regression with null lambda_
    :param y: Class labels
    :param tx: Dataset of independent variables
    :param initial_w: Initial weights
    :param max_iters: Maximum number of iterations
    :param gamma: Size of step
    :return: Weights, losses
    """
    
    masked_tx = np.nan_to_num(tx)

    w, loss = reg_logistic_regression(y, masked_tx, NULL_LAMBDA, initial_w, max_iters, gamma)

    return w, loss

