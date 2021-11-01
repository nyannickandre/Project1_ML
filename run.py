import numpy as np
from proj1_helpers import *
from implementations import *

print('Loading train data...')
DATA_TRAIN_PATH = 'data/train.csv'  # TODO: download train data and supply path here
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

print('Loading test data...')
DATA_TEST_PATH = 'data/test.csv' # TODO: download train data and supply path here
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)


#Input parameters

# Parameters for preprocessing
THRES = 20
NB_SIG = 3

#Parameters for regression
DEGREE = 8
LAMBDA_ = 0.1
JETS = 4
MAX_ITERS = 100
GAMMA = 0.3
CROSS_VALIDATIONS = 5

#Parameters for ridge
DEGREE_JET = [6, 6, 11, 9]
LAMBDA_JET = [1e-11, 1e-9, 1e-15, 1e-15]

print('')
print('------------------- RESULTS FOR TRAINING ---------------------------')
print('')
print('The {}-fold cross validation gives the following prediction for each jet:'.format(CROSS_VALIDATIONS))
print('')

# initial cleaning
tX_final, tX_mean, tX_dev, drop_colX = drop_trash(tX, THRES, NB_SIG)


# we split the data by jet values, more info in the "sep_by_jet" function
tX_j, y_j = sep_by_jet(tX_final, y, JETS)


# just a nice empty array to be filled, we will add here our predictions
y_pred = []

# empty array to be filled with the weights
weights = []

for i in range(0, JETS):
    accuracies = []
    for k in range(CROSS_VALIDATIONS):
        tx_train, tx_test, y_train, y_test = cross_val(tX_final, tX_j[i], y_j[i], DEGREE_JET[i], i, k, CROSS_VALIDATIONS)
        w, _ = ridge_regression(y_train, tx_train, LAMBDA_JET[i])
        initial_w = np.full(tx_train.shape[1], 10e-6)
        # w, _ = least_squares_GD(y_train, tx_train, initial_w, MAX_ITERS, GAMMA)
        # w, _ = least_squares_SGD(y_train, tx_train, initial_w, MAX_ITERS, GAMMA)
        # w, _ = least_squares(y_train, tx_train)
        # w, _ = logistic_regression(y_train tx_train, initial_w, MAX_ITERS, GAMMA)
        # w, _ = reg_logistic_regression(y_train, tx_train, LAMBDA_, initial_w, MAX_ITERS, GAMMA) 

        # ---------------------------------------------------------------------------------------
        y_pred_jet = predict_labels(w, tx_test)
        percentage_accuracy = sum(np.array(y_test == y_pred_jet, dtype=int)) * 100 / len(y_test)
        accuracies.append(percentage_accuracy)
  
        
    print('For jet {} : {}% accuracy'.format(i, sum(accuracies)/len(accuracies)))


# uncomment the functions below to try different implemented models

for i in range(0, JETS):#-------------------- put DEGREE_JET[i] right below for the modified RR
    tx_train, tx_test, _ = proc_jet(tX_final,tX_j[i], DEGREE_JET[i], i)
    w, _ = ridge_regression(y_j[i], tx_train, LAMBDA_JET[i])
    # initial_w = np.full(tx_train.shape[1], 10e-6)
    # w, _ = least_squares_GD(y_j[i], tx_train, initial_w, MAX_ITERS, GAMMA)
    # w, _ = least_squares_SGD(y_j[i], tx_train, initial_w, MAX_ITERS, GAMMA)
    # w, _ = least_squares(y_j[i], tx_train)
    # w, _ = logistic_regression(y_j[i], tx_train, initial_w, MAX_ITERS, GAMMA)
    # w, _ = reg_logistic_regression(y_j[i], tx_train, LAMBDA_, initial_w, MAX_ITERS, GAMMA)
    y_pred.append(predict_labels(w, tx_test))  
    weights.append(w)
    

y_pred_tot = assemble_by_jet(y_pred[0], y_pred[1], y_pred[2], y_pred[3], tX_final)
percentage_accuracy = sum(np.array(y == y_pred_tot, dtype=int)) * 100 / len(y)
print('')
print('Overall prediction of y for the train dataset is {}% accurate'.format(percentage_accuracy))
print('')
print('The confusion matrix of the training test:')
print(confusion_matrix(y, y_pred_tot))

TP = confusion_matrix(y, y_pred_tot)[0, 0]
FP = confusion_matrix(y, y_pred_tot)[0, 1]
FN = confusion_matrix(y, y_pred_tot)[1, 0]
print('The overall F1-score is',(TP/(TP + 0.5*(FP + FN))))


print('')
print('------------------- RESULTS FOR TESTING ---------------------------')


print('Processing the test set....')
#initial cleaning
tX_test_final = preproc_test(tX_test, tX_mean, tX_dev, drop_colX, NB_SIG)

# we split the data by jet values, more info in the "sep_by_jet" function
tX_testj, _ = sep_by_jet(tX_test_final, np.zeros(tX_test_final.shape[0]), JETS)


# just a nice empty array to be filled, we will add here our predictions
y_pred_test = []

print('Making the predictions...')
for i in range(0, JETS):
    tX_test_train, tX_test_test, _ = proc_jet(tX_test_final,tX_testj[i], DEGREE_JET[i], i)
    y_pred_test.append(predict_labels(weights[i], tX_test_test))  

y_pred_test_tot = assemble_by_jet(y_pred_test[0], y_pred_test[1], y_pred_test[2], y_pred_test[3], tX_test_final)


OUTPUT_PATH = 'final_submission.csv'
create_csv_submission(ids_test, y_pred_test_tot, OUTPUT_PATH)
print('The final_submission file have been writing successfully')
