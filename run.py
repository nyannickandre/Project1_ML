import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from implementations import * 

print('Loading train data...')
DATA_TRAIN_PATH = 'data/train.csv' # TODO: download train data and supply path here 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

# print('Loading test data...')
# DATA_TEST_PATH = 'data/train.csv' # TODO: download train data and supply path here 
# _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)


# ## Cleaning data
print('Start cleaning data...')


# initial_w = np.zeros(tX.shape[1])
# max_iters = 50
# gamma = 0.7
# lambda_ = 2
# w, loss = ridge_regression(y, tX[:,0], lambda_)


def drop_empty(tx,thres):
    N = tx.shape[0]
    D = tx.shape[1]
    mask = np.array(tx == -999, dtype = int)
    mask_sum = np.sum(mask,axis=0)
    perc = 100*mask_sum/N
    
    cleaner = perc < thres
    tx_clean = tx[:,cleaner]
    
    for i in range(D):
        if perc[i] > thres:
           print('The', i+1,'th column is dropped')
           
    # Maybe remove outliers: value over 2000
           
    return tx_clean




tX_clean = drop_empty(tX,40)



def sep_by_jet(tx):

    # Split data into subdatasets of different number of jet (which is the jet_num value)
    # because different number of particles means different behaviors and proportions (mass, etc.)
    # Note: The col PRI_jet_all_pt has a 0 when jet_num has a 0
    
    tX_0j = tx[tx[:,23] == 0]
    tX_1j = tx[tx[:,23] == 1]
    tX_2j = tx[tx[:,23] == 2]
    tX_3j = tx[tx[:,23] == 3]
    
    y_0j = y[tx[:,23] == 0]
    y_1j = y[tx[:,23] == 1]
    y_2j = y[tx[:,23] == 2]
    y_3j = y[tx[:,23] == 3]
    
    return tX_0j, tX_1j, tX_2j, tX_3j, y_0j, y_1j, y_2j, y_3j

print(sep_by_jet(tX))
        



















# ------------- 

#OUTPUT_PATH = 'final_submission.csv' # TODO: fill in desired name of output file for submission
#y_pred = predict_labels(weights, tX_test)
#create_csv_submission(ids_test, y_pred, OUTPUT_PATH)