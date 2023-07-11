import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import ml_tools as tools
import argparse
import math

ALPHA = 0.001
EPOCHS = 100


def compute_cost(x, y, slopes, intercept, *argv):
    """
    Computes the cost over all examples
    Args:
        x : (ndarray Shape (m,n)) data, m examples by n features
        y : (ndarray Shape (m,))  target value 
        slopes : (ndarray Shape (n,))  values of parameters of the model      
        intercept : (scalar)              value of bias parameter of the model
      *argv : unused, for compatibility with regularized version below
    Returns:
        total_cost : (scalar) cost 
    """

    m, n = x.shape
    loss = 0
    total_cost = 0
    for i in range(m):
        slopes_x_intercept = 0
        for col, j in zip(x.columns, range(n)):
            slopes_x_intercept = slopes_x_intercept + x[col] * slopes[j]
        slopes_x_intercept = slopes_x_intercept + intercept
        f_slopes_intercept_x = tools.sigmoid_(slopes_x_intercept)
        print(f_slopes_intercept_x, 'slop')
        loss = loss + (-1 * y[i] * np.log(f_slopes_intercept_x) -
                       (1 - y[i]) * np.log(1 - f_slopes_intercept_x))
    total_cost = loss / m
    return total_cost


def compute_gradient(x, y, slopes, intercept, *argv):
    """
    Computes the gradient for logistic regression 

    Args:
        x : (ndarray Shape (m,n)) data, m examples by n features
        y : (ndarray Shape (m,))  target value 
        slopes : (ndarray Shape (n,))  values of parameters of the model      
        intercept : (scalar)              value of bias parameter of the model
        *argv : unused, for compatibility with regularized version below
        Returns
        dj_dslopes : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters slopes. 
        dj_dintercept : (scalar)             The gradient of the cost w.r.t. the parameter intercept. 
    """
    m, n = x.shape
    dj_dslopes = np.zeros(slopes.shape)
    dj_dintercept = 0.
    for i in range(m):
        f_slopes_intercept_i = tools.sigmoid_(x.dot(slopes) + intercept)
        err_i = f_slopes_intercept_i - y[i]
        if i == 1:
            print(err_i, 'err_i')
        for col, j in zip(x.columns, range(n)):
            dj_dslopes[j] = dj_dslopes[j] + err_i[j] * x[col][i]
        dj_dintercept = dj_dintercept + err_i
    dj_dslopes = dj_dslopes / m
    dj_dintercept = dj_dintercept / m
    print(dj_dslopes, 'dj_dslopes')
    print(dj_dintercept, 'dj_dintercept')
    sys.exit()
    return dj_dintercept, dj_dslopes


def gradient_descent(x, y, slopes, intercept, cost_function, gradient_function, lambda_):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha

    Args:
        x :    (ndarray Shape (m, n) data, m examples by n features
        y :    (ndarray Shape (m,))  target value 
        slopes : (ndarray Shape (n,))  Initial values of parameters of the model
        intercept : (scalar)              Initial value of parameter of the model
        cost_function :              function to compute cost
        gradient_function :          function to compute gradient
        lambda_ : (scalar, float)    regularization constant

    Returns:
        slopes : (ndarray Shape (n,)) Updated values of parameters of the model after
            running gradient descent
        intercept : (scalar)                Updated value of parameter of the model after
            running gradient descent
    """

    j_history = []
    w_history = []
    for i in range(EPOCHS):

        dj_db, dj_dw = gradient_function(x, y, slopes, intercept, lambda_)
        slopes = slopes - ALPHA * dj_dw
        intercept = intercept - ALPHA * dj_db

        if i < 100000:
            cost = cost_function(x, y, slopes, intercept, lambda_)
            j_history.append(cost)
        if i % math.ceil(EPOCHS / 10) == 0 or i == (EPOCHS - 1):
            w_history.append(slopes)
            print(f"Iteration {i:4}: Cost {float(j_history[-1]):8.2f}   ")

    return slopes, intercept, j_history, w_history


def logistic_regression(x, y):
    '''The logisticRegression function performs logistic regression on the given input data and plots the
    decision boundary.

    Parameters
    ----------
    x
        The input matrix x, where each row represents a training example and each column represents a
    feature.
    y
        The parameter "y" in the logisticRegression function represents the target variable or the
    dependent variable. It is a numpy array or pandas series that contains the labels or classes for
    each data point in the dataset. In the context of logistic regression, "y" typically contains binary
    values (0 or 1

    '''
    _, n = x.shape
    intercept = 0
    slopes = np.zeros(n)
    slopes, intercept, _, _ = gradient_descent(
        x, y, slopes, intercept, compute_cost, compute_gradient, 0)
    tools.plot_decision_boundary(slopes, intercept, x, y)
    # Set the y-axis label
    plt.ylabel('Exam 2 score')
    # Set the x-axis label
    plt.xlabel('Exam 1 score')
    plt.legend(loc="upper right")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file', type=str, help='csv file')
    args = parser.parse_args()
    try:
        tools.isValidPath(args.csv_file)
    except Exception as e:
        print(e)
        sys.exit(e)

    def transform_houses(x): return 1 if x == 'Slytherin' else 0
    x, y = tools.load_data(
        args.csv_file, "Hogwarts House", transform_houses)
    logistic_regression(x, y)
    # try:
    #     x, y = tools.load_data(
    #         args.csv_file, "Hogwarts House", transformHouses)
    #     logisticRegression(x, y)
    # except Exception as e:
    #     print(f"error {e.__class__.__name__}: {e}")
    #     sys.exit(e)
