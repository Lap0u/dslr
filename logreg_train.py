import sys
import matplotlib.pyplot as plt
import numpy as np
import ml_tools as tools
import argparse

ALPHA = 0.001
EPOCHS = 1000
EPSILON = 1e-15
HOUSE_CONVERTER = {"Slytherin": 1, "Gryffindor": 2, "Ravenclaw": 3, "Hufflepuff": 4}
HOUSE_CONVERTER = {"Slytherin": 1, "Ravenclaw": 0}


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

    m, _ = x.shape
    loss = 0
    total_cost = 0
    slopes_x_intercept = x.dot(slopes) + intercept
    f_slopes_intercept_x = tools.sigmoid_(slopes_x_intercept)
    loss = -y * np.log(f_slopes_intercept_x + EPSILON) - (1 - y) * np.log(
        1 - f_slopes_intercept_x + EPSILON
    )
    total_cost = np.sum(loss) / m
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
    m, _ = x.shape
    predictions = tools.sigmoid_(np.dot(x, slopes) + intercept)
    errors = predictions - y

    dj_dslopes = (1 / m) * np.dot(x.T, errors)  # Efficient gradient for slopes
    dj_dintercept = (1 / m) * np.sum(errors)  # Gradient for intercept

    return dj_dintercept, dj_dslopes


def gradient_descent(
    x, y, slopes, intercept, cost_function, gradient_function, lambda_
):
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

    for _ in range(EPOCHS):
        cost = cost_function(x, y, slopes, intercept, lambda_)
        dj_dintercept, dj_dslopes = gradient_function(x, y, slopes, intercept, lambda_)
        intercept = intercept - ALPHA * dj_dintercept
        slopes = slopes - ALPHA * dj_dslopes
        print(f"Cost: {cost}")
    return slopes, intercept, dj_dslopes, dj_dintercept


def save_theta(slopes, intercept):
    """
    Saves the theta values to a file
    slopes : (ndarray Shape (n,)) Updated values of parameters of the model after
        running gradient descent
    intercept : (scalar)                Updated value of parameter of the model after
        running gradient descent
    """
    np.save("slopes.npy", slopes)
    np.save("intercept.npy", intercept)


def logistic_regression(x, y):
    """The logisticRegression function performs logistic regression on the given input data and plots the
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

    """
    _, n = x.shape
    slopes = np.random.rand(n)
    intercept = 1.45
    slopes, intercept, _, _ = gradient_descent(
        x, y, slopes, intercept, compute_cost, compute_gradient, 0
    )
    save_theta(slopes, intercept)


def transform_houses(x):
    return HOUSE_CONVERTER[x]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", type=str, help="csv file")
    args = parser.parse_args()
    try:
        tools.is_valid_path(args.csv_file)
    except Exception as e:
        print(e)
        sys.exit(e)

    x, y = tools.load_data(args.csv_file, "Hogwarts House", transform_houses)
    x = tools.normalize_df(x)
    logistic_regression(x, y)
    # try:
    #     x, y = tools.load_data(
    #         args.csv_file, "Hogwarts House", transformHouses)
    #     logisticRegression(x, y)
    # except Exception as e:
    #     print(f"error {e.__class__.__name__}: {e}")
    #     sys.exit(e)
