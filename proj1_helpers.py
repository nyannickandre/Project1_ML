# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


# ------------- Helpers for the project --------------------

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    masked_data = np.nan_to_num(data)
    y_pred = np.dot(masked_data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})


# ---------- Helpers from previous labs -----------

def split_data(x, y, ratio, myseed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(myseed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree + 1):
        poly = np.c_[poly, np.power(x, deg)]

    # poly = x.copy()
    # for deg in range(2, degree + 1):
    #     poly = np.hstack([poly, poly**deg])

    return poly


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.nanmean(x)
    x = x - mean_x
    std_x = np.nanstd(x)
    x = x / std_x
    return x, mean_x, std_x


def build_model_data(height, weight):
    """Form (y,tX) to get regression data in matrix form."""
    y = weight
    x = height
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx


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


# ------- Clean and preprocess -------

def stat_val(tx):
    """"
    Generate the statistics values to normalize the data in tx
    Return two vector containing respectively the mean and the standard deviation
    of each column of tx
    """
    tx_mask = np.isnan(tx)  # create a mask for all of the nan values in tx
    tx_ma = np.ma.array(tx, mask=tx_mask)  # create a masked array to avoid counting the nan when doing the calcutaions

    tx_dev = np.array(np.std(tx_ma, axis=0))
    tx_mean = np.array(np.mean(tx_ma, axis=0))
    return tx_mean, tx_dev


def drop_empty(tx, thres, nb_sig):
    """
    @ TODO : finish comments
    :param tx:
    :param thres: percentage of -999 in a column at which the column should be removed from tx
    :param nb_sig: number of standard deviation accepted, after normalizing the data if a given
    value is superior to nb_sig then it is set to nan
    :return:
    """

    # Find the jet_num column by looking for the largest amount of 1, 2 and 3
    _, col = np.where((tx == 1) | (tx == 2) | (tx == 3))
    jet_col = np.argmax(np.bincount(col))  # Find the index of the jet_num column

    # Create the jet_num vector
    jet_num = tx[:, jet_col]
    jet_num = jet_num.reshape(-1, 1)

    # Create a copy of tx and delete the jet_num column from this copy
    tx_clean = tx.copy()
    tx_clean = np.delete(tx_clean, jet_col, axis=1)

    # Get the shape of tx
    N = tx.shape[0]
    D = tx.shape[1]

    # Change all the -999 in tx_clean into nan
    tx_clean[tx_clean == -999] = np.NAN

    # Get the mean and standard deviation vectors
    tx_mean, tx_dev = stat_val(tx_clean)

    # Create tx_z as the normalization of tx_clean
    tx_z = (tx_clean - tx_mean[None, :]) / tx_dev[None, :]

    # Change the outliers in each column into nan
    # Small issues with the two following lines: create a runtime warning error
    tx_z[tx_z > nb_sig] = np.NAN
    tx_z[tx_z < -nb_sig] = np.NAN

    # Put jet_num back into tx_z in the first column
    tx_z = np.concatenate((jet_num, tx_z), axis=1)

    # Create a mask in binary to get the position of all the nan
    mask = np.array(np.isnan(tx_z), dtype=int)

    # Sum all the 1 in each column to get the percentage of nan in each column
    mask_sum = np.sum(mask, axis=0)
    perc = 100 * mask_sum / N

    # Create a mask for the column where the percentage of nan is inferior to the threshold
    cleaner = perc < thres
    tx_z = tx_z[:, cleaner]  # Delete column where there is too many nan

    # Print the column that are dropped
    for i in range(D):
        if perc[i] > thres:
            print('The {}th column is dropped'.format(i + 1))

    return tx_z


def sep_by_jet(tx, y, jets, jet_col = 0):
    # Split data into subdatasets of different number of jet (which is the jet_num value)
    # because different number of particles means different behaviors and proportions (mass, etc.)
    # Note: The col PRI_jet_all_pt has a 0 when jet_num has a 0

    tX_j = []
    y_j = []

    for i in range(0, jets):
        tX_j.append(tx[tx[:, jet_col] == i])
        y_j.append(y[tx[:, jet_col] == i])

    return tX_j, y_j


def assemble_by_jet(y_0, y_1, y_2, y_3, tx):
    N = tx.shape[0]
    y = np.zeros(N)
    jet_num = tx[:, 0]
    count_0 = 0;
    count_1 = 0;
    count_2 = 0;
    count_3 = 0;

    for i in range(N):
        x = jet_num[i]
        if x == 0:
            y[i] = y_0[count_0]
            count_0 = count_0 + 1
        elif x == 1:
            y[i] = y_1[count_1]
            count_1 = count_1 + 1
        elif x == 2:
            y[i] = y_2[count_2]
            count_2 = count_2 + 1
        elif x == 3:
            y[i] = y_3[count_3]
            count_3 = count_3 + 1

    return y


def proc_jet(tx, degree, num_jet, tx_jet):
    """
    Processes data by selecting those corresponding to the right jet and creating polynomials for the future regression.
    :param tx: (Multi dimensional array)
    :param degree: (int)
    :param num_jet: (int)
    :param tx_jet:
    :return: (tuple) Processed train data, processed test data, @ TODO : what is the 3rd ?
    """

    jet_col = 0

    idx_test = (tx[:, jet_col] == num_jet)
    print('------------------------------------------------------')
    print(idx_test)

    tx_test = tx[idx_test]

    # Delete the Jet_col column
    # tx_test_jet = np.delete(tx_test, [jet_col], 1)

    # Build a polynomial function of given degree
    tx_train_poly = build_poly(tx_jet, degree)
    print('------------------------------------------------------')

    tx_test_poly = build_poly(tx_test, degree)

    # Standardization of the polynome
    tx_train_stand, mean_tx_train, std_tx_train = standardize(tx_train_poly)
    tx_test_stand = (tx_test_poly - mean_tx_train) / std_tx_train

    # Offset to have the intercept
    tx_off = np.insert(tx_train_stand, 0, np.ones(tx_jet.shape[0]), axis=1)
    tx_test_off = np.insert(tx_test_stand, 0, np.ones(tx_test_stand.shape[0]), axis=1)

    return tx_off, tx_test_off, idx_test


# ------- Test -------

def confusion_matrix(true, pred):
    """
    Computes the confusion matrix, as ratios
    :param true: (Array, either 1 or -1) True results of wether an element is a boson or not.
    :param pred: (Array, either 1 or -1) Predicted results
    :return: (2 x 2 array) Confusion matrix. Top left is (true, true)
    """

    comparison = np.column_stack((true, pred))
    confusion = np.zeros((2, 2))

    for elem in comparison:
        # true positives
        if np.array_equal(elem, np.array([1, 1])):
            confusion[0][0] += 1
        # false positives
        elif np.array_equal(elem, np.array([-1, 1])):
            confusion[0, 1] += 1
        # false negative
        elif np.array_equal(elem, np.array([1, -1])):
            confusion[1, 0] += 1
        else:
            confusion[1, 1] += 1

    return confusion / len(true)




