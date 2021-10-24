import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from implementations import *

DATA_TRAIN_PATH = 'data/train.csv'  # TODO: download train data and supply path here
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

DEGREE = 6
LAMBDA_ = 1e-6
JETS = 4

# print('Loading test data...')
# DATA_TEST_PATH = 'data/test.csv' # TODO: download train data and supply path here
# _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

# initial cleaning
tX_final = drop_trash(tX, 40, 3)

# we split the data by jet values, more info in the "sep_by_jet" function
tX_j, y_j = sep_by_jet(tX_final, y, JETS)

# just a nice empty array to be filled, we will add here our predictions
y_pred = []

for i in range(0, JETS):
    tx_train, tx_test, idx_test = proc_jet(tX_final, DEGREE, i, tX_j[i])
    w, _ = ridge_regression(y_j[i], tx_train, LAMBDA_)
    y_pred.append(predict_labels(w, tx_test))

# assembles predicted class labels
y_pred_tot = assemble_by_jet(y_pred[0], y_pred[1], y_pred[2], y_pred[3], tX_final)

print(y_pred_tot[:20])

percentage_accuracy = sum(np.array(y == y_pred_tot, dtype=int)) * 100 / len(y)
print('-------------------------------------------------------------------')
print('The prediction is {}% accurate'.format(percentage_accuracy))

print(confusion_matrix(y, y_pred_tot))

# -------------

# OUTPUT_PATH = 'final_submission.csv' # TODO: fill in desired name of output file for submission
# y_pred = predict_labels(weights, tX_test)
# create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

# TODO : Delete the "prints" used for debugging in all files
