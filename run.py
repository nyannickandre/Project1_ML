import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from implementations import * 

print('Loading train data...')
DATA_TRAIN_PATH = 'data/train.csv' # TODO: download train data and supply path here 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

# print('Loading test data...')
# DATA_TEST_PATH = 'data/test.csv' # TODO: download train data and supply path here 
# _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)


# ## Cleaning data
print('Start cleaning data...')

tX_final = drop_empty(tX,40,3)

tX_0j, tX_1j, tX_2j, tX_3j, y_0j, y_1j, y_2j, y_3j = sep_by_jet(tX_final,y) 

degree = 6

tx_train_0j, tx_test_0j, idx_test_0j = proc_jet(tX_0j, degree, 0, tX_0j)

lambda_ = 1
w_0j, _ = ridge_regression(y_0j, tx_train_0j, lambda_)

y_pred0 = predict_labels(w_0j, tx_test_0j)


# PROCESS DATA

















# ------------- 

#OUTPUT_PATH = 'final_submission.csv' # TODO: fill in desired name of output file for submission
#y_pred = predict_labels(weights, tX_test)
#create_csv_submission(ids_test, y_pred, OUTPUT_PATH)