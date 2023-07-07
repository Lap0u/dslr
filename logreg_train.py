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


def compute_cost(X, y, w, b, *argv):
    """
    Computes the cost over all examples
    Args:
        X : (ndarray Shape (m,n)) data, m examples by n features
        y : (ndarray Shape (m,))  target value 
        w : (ndarray Shape (n,))  values of parameters of the model      
        b : (scalar)              value of bias parameter of the model
      *argv : unused, for compatibility with regularized version below
    Returns:
        total_cost : (scalar) cost 
    """

    m, n = X.shape
    loss = 0
    total_cost = 0
    for i in range(m):
        wx_b = 0
        for j in range(n):
            wx_b = wx_b + X[i][j] * w[j]
        wx_b = wx_b + b
        f_wb_X = tools.sigmoid_(wx_b)
        loss = loss + (-1 * y[i] * np.log(f_wb_X) -
                       (1 - y[i]) * np.log(1 - f_wb_X))
    total_cost = loss / m
    return total_cost


def compute_gradient(X, y, w, b, *argv):
    """
    Computes the gradient for logistic regression 

    Args:
        X : (ndarray Shape (m,n)) data, m examples by n features
        y : (ndarray Shape (m,))  target value 
        w : (ndarray Shape (n,))  values of parameters of the model      
        b : (scalar)              value of bias parameter of the model
        *argv : unused, for compatibility with regularized version below
        Returns
        dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 
        dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b. 
    """
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    for i in range(m):
        f_wb_i = tools.sigmoid_(np.dot(X[i], w) + b)
        err_i = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i][j]
        dj_db = dj_db + err_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_db, dj_dw


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, lambda_):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha

    Args:
        X :    (ndarray Shape (m, n) data, m examples by n features
        y :    (ndarray Shape (m,))  target value 
        w_in : (ndarray Shape (n,))  Initial values of parameters of the model
        b_in : (scalar)              Initial value of parameter of the model
        cost_function :              function to compute cost
        gradient_function :          function to compute gradient
        lambda_ : (scalar, float)    regularization constant

    Returns:
        w : (ndarray Shape (n,)) Updated values of parameters of the model after
            running gradient descent
        b : (scalar)                Updated value of parameter of the model after
            running gradient descent
    """

    J_history = []
    w_history = []

    for i in range(EPOCHS):

        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)

        w_in = w_in - ALPHA * dj_dw
        b_in = b_in - ALPHA * dj_db

        if i < 100000:
            cost = cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        if i % math.ceil(EPOCHS / 10) == 0 or i == (EPOCHS - 1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")

    return w_in, b_in, J_history, w_history


def logisticRegression(X, y):
    m, n = X.shape
    np.random.seed(1)
    slopes = 0.01 * (np.random.rand(2) - 0.5)
    intercept = -8
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
        sys.exit(e)
    X, y = tools.load_data(args.csv_file)
    logisticRegression(X, y)
