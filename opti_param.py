import numpy as np
from proj1_helpers import *
from implementations import *

DATA_TRAIN_PATH = 'data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

# Parameters for preprocessing
THRES = 20
NB_SIG = 3


JETS = 4
CROSS_VALIDATIONS = 5

# found after doing an additional loop on LAMBDA_JET in np.geomspace(1,1e-15,16)
LAMBDA_JET = [1e-11, 1e-9, 1e-15, 1e-15]

# initial cleaning
tX_final, tX_mean, tX_dev, drop_colX = drop_trash(tX, THRES, NB_SIG)


# we split the data by jet values, more info in the "sep_by_jet" function
tX_j, y_j = sep_by_jet(tX_final, y, JETS)


# arrays to fill to plot the optimized degrees, and lambdas, 
# in order to get the best combination of degree and lambda
# opti0 = []
# opti1 = []
# opti2 = []
# opti3 = []

# arrays to fill to be able to get the plot of our report
deg_graph0 = []
deg_graph1 = []
deg_graph2 = []
deg_graph3 = []


for DEGREE_JET in range(1,13):
    for i in range(0, JETS):
        accuracies = []
        F1 = []
        for k in range(CROSS_VALIDATIONS):
            tx_train, tx_test, y_train, y_test = cross_val(tX_final, tX_j[i], y_j[i], DEGREE_JET, i, k, CROSS_VALIDATIONS)
            w, _ = ridge_regression(y_train, tx_train, LAMBDA_JET[i])
            # initial_w = np.full(tx_train.shape[1], 10e-6)
            # w, _ = least_squares_GD(y_train, tx_train, initial_w, MAX_ITERS, GAMMA)
            # w, _ = least_squares_SGD(y_train, tx_train, initial_w, MAX_ITERS, GAMMA)
            # w, _ = least_squares(y_train, tx_train)
            # w, _ = logistic_regression(y_train, tx_train, initial_w, MAX_ITERS, GAMMA)
            # w, _ = reg_logistic_regression(y_train, tx_train, LAMBDA_, initial_w, MAX_ITERS, GAMMA)
 
            y_pred_jet = predict_labels(w, tx_test)
            percentage_accuracy = sum(np.array(y_test == y_pred_jet, dtype=int)) / len(y_test)
            accuracies.append(percentage_accuracy)
            
            TP = confusion_matrix(y_test, y_pred_jet)[0, 0]
            FP = confusion_matrix(y_test, y_pred_jet)[0, 1]
            FN = confusion_matrix(y_test, y_pred_jet)[1, 0]
            
            F1.append(TP/(TP + 0.5*(FP + FN)))
                    
        
        if i == 0:
            # opti0.append([DEGREE_JET,LAMBDA_JET,sum(accuracies)/len(accuracies)])
            deg_graph0.append([DEGREE_JET,sum(accuracies)/len(accuracies),sum(F1)/len(F1)])
            
        elif i == 1:
            # opti1.append([DEGREE_JET,LAMBDA_JET,sum(accuracies)/len(accuracies)])
            deg_graph1.append([DEGREE_JET,sum(accuracies)/len(accuracies),sum(F1)/len(F1)])
            
        elif i == 2:
            # opti2.append([DEGREE_JET,LAMBDA_JET,sum(accuracies)/len(accuracies)])
            deg_graph2.append([DEGREE_JET,sum(accuracies)/len(accuracies),sum(F1)/len(F1)])
            
        elif i == 3:
            # opti3.append([DEGREE_JET,LAMBDA_JET,sum(accuracies)/len(accuracies)])
            deg_graph3.append([DEGREE_JET,sum(accuracies)/len(accuracies),sum(F1)/len(F1)])
            
    
        print('For jet {} with degree {}, the prediction has {} acc. and {} F1-score'.format(i, DEGREE_JET, sum(accuracies)/len(accuracies), sum(F1)/len(F1)))


# import matplotlib to be able to plot our results and put it on the report
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 2, figsize=(12,8))
axs[0, 0].plot([row[0] for row in deg_graph0],[row[1] for row in deg_graph0],'g',label='Accuracy')
axs[0, 0].plot([row[0] for row in deg_graph0],[row[2] for row in deg_graph0],'b',label='F1-score')
axs[0, 0].set_title("Jet_num = 0")
axs[0, 0].legend()
axs[0, 0].grid()
axs[0, 1].plot([row[0] for row in deg_graph1],[row[1] for row in deg_graph1],'g',label='Accuracy')
axs[0, 1].plot([row[0] for row in deg_graph1],[row[2] for row in deg_graph1],'b',label='F1-score')
axs[0, 1].set_title("Jet_num = 1")
axs[0, 1].legend()
axs[0, 1].grid()
axs[1, 0].plot([row[0] for row in deg_graph2],[row[1] for row in deg_graph2],'g',label='Accuracy')
axs[1, 0].plot([row[0] for row in deg_graph2],[row[2] for row in deg_graph2],'b',label='F1-score')
axs[1, 0].set_title("Jet_num = 2")
axs[1, 0].legend()
axs[1, 0].set_xlabel('Degree')
axs[1, 0].grid()
axs[1, 1].plot([row[0] for row in deg_graph3],[row[1] for row in deg_graph3],'g',label='Accuracy')
axs[1, 1].plot([row[0] for row in deg_graph3],[row[2] for row in deg_graph3],'b',label='F1-score')
axs[1, 1].set_title("Jet_num = 3")
axs[1, 1].legend()
axs[1, 1].set_xlabel('Degree')
axs[1, 1].grid()
plt.show()


