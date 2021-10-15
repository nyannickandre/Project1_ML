import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from implementations import *

DATA_TRAIN_PATH = 'data/train.csv' # TODO: download train data and supply path here 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

# print('Loading test data...')
# DATA_TEST_PATH = 'data/test.csv' # TODO: download train data and supply path here 
# _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)


tX_final = drop_empty(tX,40,3)

tX_0j, tX_1j, tX_2j, tX_3j, y_0j, y_1j, y_2j, y_3j = sep_by_jet(tX_final,y)

degree = 6

tx_train_0j, tx_test_0j, idx_test_0j = proc_jet(tX_final, degree, 0, tX_0j)
tx_train_1j, tx_test_1j, idx_test_1j = proc_jet(tX_final, degree, 1, tX_1j)
tx_train_2j, tx_test_2j, idx_test_2j = proc_jet(tX_final, degree, 2, tX_2j)
tx_train_3j, tx_test_3j, idx_test_3j = proc_jet(tX_final, degree, 3, tX_3j)

lambda_ = 1e-6
w_0j, _ = ridge_regression(y_0j, tx_train_0j, lambda_)
w_1j, _ = ridge_regression(y_1j, tx_train_1j, lambda_)
w_2j, _ = ridge_regression(y_2j, tx_train_2j, lambda_)
w_3j, _ = ridge_regression(y_3j, tx_train_3j, lambda_)


y_pred0 = predict_labels(w_0j, tx_test_0j)
y_pred1 = predict_labels(w_1j, tx_test_1j)
y_pred2 = predict_labels(w_2j, tx_test_2j)
y_pred3 = predict_labels(w_3j, tx_test_3j)

y_pred_tot = assemble_by_jet(y_pred0,y_pred1,y_pred2,y_pred3,tX_final)

percentage_accuracy = sum(np.array(y == y_pred_tot, dtype=int))*100/len(y)
print('-------------------------------------------------------------------')
print('The prediction is {}% accurate'.format(percentage_accuracy))


















# ------------- 

#OUTPUT_PATH = 'final_submission.csv' # TODO: fill in desired name of output file for submission
#y_pred = predict_labels(weights, tX_test)
#create_csv_submission(ids_test, y_pred, OUTPUT_PATH)