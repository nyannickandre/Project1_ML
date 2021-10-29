import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from implementations import *

DATA_TRAIN_PATH = 'data/train.csv'  # TODO: download train data and supply path here
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

DEGREE = 6
LAMBDA_ = 1e-15
JETS = 4
MAX_ITERS = 100
GAMMA = 0.3
CROSS_VALIDATIONS = 5

DEGREE_JET = [3, 6, 4, 4]
LAMBDA_JET = [1e-10, 1e-11, 1e-8, 1e-15]

# print('Loading test data...')
# DATA_TEST_PATH = 'data/test.csv' # TODO: download train data and supply path here
# _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

# initial cleaning
tX_final = drop_trash(tX, 40, 3)


# we split the data by jet values, more info in the "sep_by_jet" function
tX_j, y_j = sep_by_jet(tX_final, y, JETS)


# just a nice empty array to be filled, we will add here our predictions
#y_pred = []

for i in range(0, JETS):
    accuracies = []
    for k in range(CROSS_VALIDATIONS):
        tx_train, tx_test, y_train, y_test = proc_jet(tX_final, tX_j[i], y_j[i], DEGREE_JET[i], i, k, CROSS_VALIDATIONS)
        w, _ = ridge_regression(y_train, tx_train, LAMBDA_JET[i])
        # initial_w = np.full(tx_train.shape[1], 10e-6)
        # w, _ = least_squares_GD(y_j[i], tx_train, initial_w, MAX_ITERS, GAMMA)
        # w, _ = least_squares_SGD(y_j[i], tx_train, initial_w, MAX_ITERS, GAMMA)
        # w, _ = least_squares(y_j[i], tx_train)
        # w, _ = logistic_regression(y_j[i], tx_train, initial_w, MAX_ITERS, GAMMA) #TODO : why do we have overflows ?
        # w, _ = reg_logistic_regression(y_j[i], tx_train, LAMBDA_, initial_w, MAX_ITERS, GAMMA) #TODO : why do we have overflows ?
        # Using RR as the best function and tuning one lambda per jet_num
        # w, _ = ridge_regression(y_j[i], tx_train, LAMBDA_JET[i])

        # ---------------------------------------------------------------------------------------
        y_pred_jet = predict_labels(w, tx_test)
        percentage_accuracy = sum(np.array(y_test == y_pred_jet, dtype=int)) * 100 / len(y_test)
        accuracies.append(percentage_accuracy)

    # y_pred.append(y_pred_jet)

    print('For jet {} the prediction is {}% accurate'.format(i+1, sum(accuracies)/len(accuracies)))


# TODO : read the comment below <3
# assembles predicted class labels
# This will probably not work anymore, but we don't really care. we optimize the lambda using cross validation
# and once found, we use this lambda on the whole jet, using directly the run.py function
# so basically, once you found the best lambda parameters, just use them in the run.py and compute the whole
# output there. Cheers!

# y_pred_tot = assemble_by_jet(y_pred[0], y_pred[1], y_pred[2], y_pred[3], tX_final)
#percentage_accuracy = sum(np.array(y == y_pred_tot, dtype=int)) * 100 / len(y)
#print('-------------------------------------------------------------------')
#print('The prediction is {}% accurate'.format(percentage_accuracy))



#print(confusion_matrix(y, y_pred_tot))
# Find F1-score using TP, TN, FP, FN
# TP, FP, FN, TN = confusion_matrix(y, y_pred_tot)
# F1 = TP/(TP + 0.5*(FP + FN))
# print('The F1-score is {}%'.format(F1))

# -------------

# OUTPUT_PATH = 'final_submission.csv' # TODO: fill in desired name of output file for submission
# y_pred = predict_labels(weights, tX_test)
# create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

# TODO : Delete the "prints" used for debugging in all files

