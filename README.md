# Project1_ML: Higgs Boson Recognition

Authors: Yannick Neypatraiky, Stefan Rotarus & Paul Habert

## Description
The **Finding the Higgs Boson** challenge is a popular machine learning competition. Here you can see the code used by our team in the context of the EPFL Machine Learning course online competition on [Aicrowd](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs).

### Dataset desciption:
The  codes implemented here allow to create a submission file for the competition platform, in the same format as the `sample-submission.csv` file that is provided.

Two files named `train.csv` and `test.csv` can be taken from [Aicrowd](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs), or by unzipping the file `data.zip`, which are respectively the training set of 250'000 events and 32 columns (ID column, label column and finally the 30 features columns) and the test set of almost 570'000 events, with the same types of columns at the exception that the label needs to be predicted.

The label column is associated with a vector *y* while the 30 features forms a matrix *tX* and the machine learning training realized allow to estimate the vector of weights *w* such that *y ~ tX w*. 

### Project Files:
All of the implementations carried out in this project were done on Python, only using the ***numpy*** library and the code is comprised with four main python files:
- `run.py`: the main file of the project which upon running will create (or update) a file named `final_submission.csv` containing the ID column and label prediction for the test set. Submitting this file on the competition platform will yield *Categorical Accuracy* of 0.819 and a *F1-Score* of 0.720. 

- `implementations.py`: the file containing the various ML methods implemented and tested on the training dataset.
- `proj1_helpers.py`: the file containing the additional methods that helps preprocess and clean the datasets and/or improve and tune the model 
- `opti_param.py`: an additional file that follows that iterates on the training dataset to find the optimal parameters for the methods used in `run.py`

A more thorough description of each file is given below.

## How to run the program

In order to run the program, only the file `run.py` is required to be opened. The file structure is as follows:

*   Loading of the datasets (Train and Test) in their vector/matrix form using the helpers methods 
*   Input the parameters to be used: 
    -   The parameters for preprocessing: 

    > THRES: the threshold at which a column of the train dataset is removed if the proportion of missing values + outliers is superior to that threshold

    > NB_SIG:  a specified number of standard deviation should given after standardization of the datasets to define the value at which a value is considered as an outlier

    -   Parameters for regression: used for all basic methods in `implementations.py` 
    > LAMBDA_ , MAX_ITERS , GAMMA: basic input parameters required to use the regression methods 
    
    > CROSS_VALIDATIONS: number of k-folds used for the cross validation
    
    > JETS = 4: number of jets found in the jet_num column feature of the dataset
    
    -   Parameters for the modified ridge regression:
    >   DEGREE_JET, LAMBDA_JET: vectors of 4 value containing respectively the optimal value of degree and lambda for the modified ridge regression realized on each jet

* Use of drop_trash method to clean the train dataset 
* Use of sep_by_jet to split the dataset by jet values 
* Use of a loop on the number of jets JETS to obtain the weights for each sub-dataset:
    -   For each jet i, the dataset are processed for the regression. 
  
            tx_train, tx_test, _ = proc_jet(tX_final,tX_j[i], DEGREE_JET[i], i)

        The current parameter DEGREE_JET[i] is the optimal degree for the modified ridge regression, and should be replaced by DEGREE if another method is used
        
    -   Currently the weights are obtained with a basic ridge_regression applied on each sub-dataset using the following line of code: 

                w, _ = ridge_regression(y_j[i], tx_train, LAMBDA_JET[i])
                
         In the case another method is used, the user can comment the line above and uncomment one of the commented lines within the loop to call one of the other ML methods implemented.
        Note that the following line needs to be uncommented for all methods requiring initial weights as input. 
        
            initial_w = np.full(tx_train.shape[1], 10e-6)
- The weight are then store in array and used in the the predict_labels method to make a prediction of the labels for each jet. The prediction are then reassembled as a single vector by using assemble_by_jet.

- The code will then display the accuracy, confusion matrix and F1-score of the model on the train dataset.  

- The test dataset labels are then obtained by preprocessing the dataset, then using the weigth calculated in the training part. Finally the IDs and labels are written into a csv file that can be submitted on the comprtition platform.

**NOTE**: The `opti_param.py` file follows the structure of `run.py` but iterates on the degree and/or the lambdas to find the optimal paramaters by finding the combination that produced the best accuracy.

## Methods implemented 
In the following, an overview of all the methods used in the project is given. 

### Machine Learning methods

The following basic ML methods, developed during the [course labs](https://github.com/epfml/ML_course) were implemented in the `implementations.py` file:  
| Functions      | Description |
| ----------- | ----------- |
| least\_squares\_GD(*y*, *tx*, *initial\_w*, *max\_iters*, *gamma*)      | Linear regression using gradient descent (GD) |
| least\_squares\_SGD(*y*, *tx*, *initial\_w*, *max\_iters*, *gamma*)  | Linear regression using stochastic GD  (SGD)|
| least\_squares(*y*, *tx*)  | Least squares regression using normal equations  |
| ridge\_regression(*y*, *tx*, *lambda\_* )   | Ridge regression using normal equations  |
| logistic\_regression(*y*, *tx*, *initial\_w*, *max\_iters*, *gamma*) | Logistic regression using GD |
| reg\_logistic\_regression(*y*, *tx*, *lambda_*, *initial\_w*, *max\_iters*, *gamma*) | Regularized logistic regression using GD |

The different inputs those functions takes are described as follows:
- *y*: label vector
- *tx*: Dataset of independent variables represented as a matrix 
- *lambda\_*: Weight of squared weights element for computing loss
- *initial\_w*: Initial vector of weights
- *max\_iters*: Maximum number of iterations
- *gamma*: Size of step

All of these functions returns as output the tuple (*w*, *loss*), which are respectively the last weight vector of the method, and the corresponding loss value (cost function). 

In order to simplify the algorithms, intermediary methods were implemented to compute the error, the gradient and the loss associated to the regression methods. Additionally, a function to compute the sigmoid was written for the logistic regression and as such the error, gradient and loss computations were implemented separately for the two logistic regression methods.

### Helpers methods 

The `proj1_helpers.py` file is divided in four parts: 

#### Helpers for the project 
This part is comprised with helpers functions provided at the beginning of the project:
| Functions      | Description |
| ----------- | ----------- |
|load_csv_data| Loads csv dataset |
|predict_labels| Generates class predictions given the weights and a test data matrix |
|create_csv_submission|Creates the csv submission file with the predictions |

#### Helpers from previous labs

This part which contains helpers functions used in previous labs exercices that were deemed significant for the ML model used in this project: 
| Functions      | Description |
| ----------- | ----------- |
|split_data | Split the dataset based on the split ratio |
| build_poly| Return polynomial basis functions for input data up to specified degree|
|standardize| Standardize the original data set|
| build_model_data|Form (y,tX) to get regression data in vector/matrix form.|
|batch_iter| Generate a minibatch iterator for a dataset.|

#### Clean and preprocess
In this part, the methods were entirely created by our team to tune the model: 
| Functions      | Description |
| ----------- | ----------- |
| stat_val | Generates the mean and standard deviation to standardized the dataset |
|drop_trash | Cleans dataset of outliers and strange columns |
| sep_by_jet|  Split data into sub-datasets depending on their jet_num value |
| assemble_by_jet | Reassembles the class label prediction of all jets together |
| randomize | Returns a randomized set of indexes the length of the dataset rows |
| split_train_test | Split the train set according to the number of k-folds for the cross validation |
| proc_jet | Processes data by selecting those corresponding to the right jet and creating polynomials for the future regression. |
|  cross_val | Cross validating the data by using proc_jet |

#### Test
In this part, the methods implementesd are those that allow to check the accuracy of the model training and allow to preprocess the test set:
| Functions      | Description |
| ----------- | ----------- |
| confusion_matrix | Computes the 2 x 2 confusion matrix for the training dataset |
| preproc_test |  Cleans test dataset to make it match with the train dataset. Same standardization, columns dropped and outliers removed  |




  



