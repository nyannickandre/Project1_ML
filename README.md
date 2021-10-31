# Project1_ML: Higgs Boson Recognition

Authors: Yannick Neypatraiky, Stefan Rotarus & Paul Habert

## Description
The **Finding the Higgs Boson** challenge is a popular machine learning competition. Here you can see the code used by our team in the context of the EPFL Machine Learning course online competition on [Aicrowd](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs).

### Dataset desciption:
The  codes implemented here allow to create a submission file for the competition platform, in the same format as the `sample-submission.csv` file that is provided.

Two files named `train.csv` and `test.csv` can be taken from [Aicrowd](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs)  which are respectively the training set of 250'000 events and 32 columns (ID column, label column and finally the 30 features columns) and the test set of almost 570'000 events, with the same types of columns at the exception that the label needs to be predicted.

The label column is associated with a vector *y* while the 30 features forms a matrix *tX* and the machine learning training realized allow to estimate the vector of weights *w* such that *y ~ tX w*. 

### Project Files:
All of the implementations carried out in this project were done on Python, only using the ***numpy*** library and the code is comprised with four main python files:
- `run.py`: the main file of the project which upon running will create (or update) a file named `final_submission.csv` containing the ID column and label prediction for the test set. Submitting this file on the competition platform will yield *Categorical Accuracy* of 0.819 and a *F1-Score* of 0.720. 

- `implementations.py`: the file containing the various ML methods implemented and tested on the training dataset.
- `proj1_helpers.py`: the file containing the additional methods that helps preprocess and clean the datasets and/or improve and tune the model see [below](#id)
- `opti_param.py`: an additional file that follows that iterates on the training dataset to find the optimal parameters for the methods used in `run.py`

A more thorough description of each file is given below.

## Machine Learning methods

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

## Helpers methods {#id}

The `proj1_helpers.py` file is divided in three parts: 

#### Helpers for the project 
This part comprised with the following functions provided at the beginning of the project:
>
        load_csv_data(data_path, sub_sample=False):
            """ Loads csv dataset """
            return: binary class label, dataset of independent variables, event ids
>
        predict_labels(weights, data, sigma = False):
            """ Generates class predictions given weights, and a test data matrix """
            return: Predicted class labels
>
        create_csv_submission(ids, y_pred, name):
            """ Creates an output file with predictions in CSV format for submission """
            return: None
#### Helpers from previous labs

This part which contains helpers functions used in previous labs exercices that were deemed significant for the ML model used in this project: 
>
    split_data(x, y, ratio, myseed=1):
        """split the dataset based on the split ratio."""
>   
    build_poly(x, degree):
        """polynomial basis functions for input data x, for j=0 up to j=degree."""
>
    standardize(x):
        """Standardize the original data set."""

>
    build_model_data(height, weight):
        """Form (y,tX) to get regression data in matrix form."""
>   
    batch_iter(y, tx, batch_size=1, num_batches=1, shuffle=True):
        """ Generate a minibatch iterator for a dataset."""
        
#### Clean and preprocess
    

## TODOs and perspectives

  



