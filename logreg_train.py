import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import ml_tools as tools
import argparse
import math

ALPHA = 0.001
EPOCHS = 10000


def compute_cost(X, y, slopes, intercept, *argv):
    """
    Computes the cost over all examples
    Args:
        X : (ndarray Shape (m,n)) data, m examples by n features
        y : (ndarray Shape (m,))  target value 
        slopes : (ndarray Shape (n,))  values of parameters of the model      
        intercept : (scalar)              value of bias parameter of the model
      *argv : unused, for compatibility with regularized version below
    Returns:
        total_cost : (scalar) cost 
    """

    m, n = X.shape
    loss = 0
    total_cost = 0
    for i in range(m):
        slopes_X_intercept = 0
        for j in range(n):
            slopes_X_intercept = slopes_X_intercept + X[i][j] * slopes[j]
        slopes_X_intercept = slopes_X_intercept + intercept
        f_slopes_intercept_X = tools.sigmoid_(slopes_X_intercept)
        loss = loss + (-1 * y[i] * np.log(f_slopes_intercept_X) -
                       (1 - y[i]) * np.log(1 - f_slopes_intercept_X))
    total_cost = loss / m
    return total_cost


def compute_gradient(X, y, slopes, intercept, *argv):
    """
    Computes the gradient for logistic regression 

    Args:
        X : (ndarray Shape (m,n)) data, m examples by n features
        y : (ndarray Shape (m,))  target value 
        slopes : (ndarray Shape (n,))  values of parameters of the model      
        intercept : (scalar)              value of bias parameter of the model
        *argv : unused, for compatibility with regularized version below
        Returns
        dj_dslopes : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters slopes. 
        dj_dintercept : (scalar)             The gradient of the cost w.r.t. the parameter intercept. 
    """
    print('in compute_gradient')
    m, n = X.shape
    dj_dslopes = np.zeros(slopes.shape)
    dj_dintercept = 0.
    print(X)
    print(m)
    print(n)
    for i in range(m):
        f_slopes_intercept_i = tools.sigmoid_(X.dot(slopes) + intercept)
        err_i = f_slopes_intercept_i - y[i]
        for j in range(n):
            dj_dslopes[j] = dj_dslopes[j] + err_i * X[i][j]
        dj_dintercept = dj_dintercept + err_i
    dj_dslopes = dj_dslopes / m
    dj_dintercept = dj_dintercept / m
    return dj_dintercept, dj_dslopes


def gradient_descent(X, y, slopes, intercept, cost_function, gradient_function, lambda_):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha

    Args:
        X :    (ndarray Shape (m, n) data, m examples by n features
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

    J_history = []
    w_history = []
    print('before gradient descent')
    for i in range(EPOCHS):

        dj_db, dj_dw = gradient_function(X, y, slopes, intercept, lambda_)

        slopes = slopes - ALPHA * dj_dw
        intercept = intercept - ALPHA * dj_db

        if i < 100000:
            cost = cost_function(X, y, slopes, intercept, lambda_)
            J_history.append(cost)

        if i % math.ceil(EPOCHS / 10) == 0 or i == (EPOCHS - 1):
            w_history.append(slopes)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")

    return slopes, intercept, J_history, w_history


def logisticRegression(X, y):
    m, n = X.shape
    np.random.seed(1)
    intercept = 0
    slopes = np.zeros(n)
    slopes, intercept, J_history, _ = gradient_descent(
        X, y, slopes, intercept, compute_cost, compute_gradient, 0)
    tools.plot_decision_boundary(slopes, intercept, X, y)
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

    def transformHouses(x): return 1 if x == 'Slytherin' else 0
    X, y = tools.load_data(
        args.csv_file, "Hogwarts House", transformHouses)
    logisticRegression(X, y)
    # try:
    #     X, y = tools.load_data(
    #         args.csv_file, "Hogwarts House", transformHouses)
    #     logisticRegression(X, y)
    # except Exception as e:
    #     print(f"error {e.__class__.__name__}: {e}")
    #     sys.exit(e)
