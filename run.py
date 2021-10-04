import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from implementations import * 

DATA_TRAIN_PATH = 'data/train.csv' # TODO: download train data and supply path here 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)


DATA_TEST_PATH = 'data/train.csv' # TODO: download train data and supply path here 
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)




initial_w = np.zeros(tX.shape[1])
max_iters = 50
gamma = 0.7
lambda_ = 2
w, loss = ridge_regression(y, tX, lambda_)



















# ------------- 

#OUTPUT_PATH = 'final_submission.csv' # TODO: fill in desired name of output file for submission
#y_pred = predict_labels(weights, tX_test)
#create_csv_submission(ids_test, y_pred, OUTPUT_PATH)