import sys
import matplotlib.pyplot as plt
import numpy as np
import ml_tools as tools
import argparse

ALPHA = 0.01
EPOCHS = 10000
EPSILON = 1e-15
HOUSES = ["Slytherin", "Gryffindor", "Ravenclaw", "Hufflepuff"]


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

    dj_dslopes = (1 / m) * np.dot(x.T, errors)
    dj_dintercept = (1 / m) * np.sum(errors)

    return dj_dintercept, dj_dslopes


def transform_houses(y, house):
    return np.where(y == house, 1, 0)


def gradient_descent(
    x,
    y,
    slopes_s,
    intercept_s,
    cost_function,
    gradient_function,
    lambda_,
    cost_show,
    mini_batch,
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
    cost_history = []
    intercept = []
    slopes = []
    # use mini batch here
    for j in range(len(HOUSES)):
        cost_history.append([])
        intercept.append([])
        slopes.append([])
        intercept[j] = intercept_s
        slopes[j] = slopes_s
        transformed_y = transform_houses(y, HOUSES[j])
        for i in range(EPOCHS):
            if cost_show:
                cost = cost_function(x, transformed_y, slopes[j], intercept[j], lambda_)
                if i % 100 == 0:
                    print(f"Epoch: {i} <-> Cost: {cost}")
                cost_history[j].append(cost)
            dj_dintercept, dj_dslopes = gradient_function(
                x, transformed_y, slopes[j], intercept[j], lambda_
            )
            intercept[j] = intercept[j] - ALPHA * dj_dintercept
            slopes[j] = slopes[j] - ALPHA * dj_dslopes
    return slopes, intercept, cost_history


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


def show_cost(cost_history):
    colors = ["red", "blue", "green", "magenta"]
    fig, axs = plt.subplots(2, 2)
    fig.suptitle("Cost function")
    for i in range(len(HOUSES)):
        axs[i // 2, i % 2].plot(cost_history[i], color=colors[i])
        axs[i // 2, i % 2].set_title(f"{HOUSES[i]} vs all")
    fig.tight_layout(pad=3.0)
    plt.show()


def compute_accuracy(x, y, slopes, intercept):
    predictions = []
    for j in range(len(HOUSES)):
        predictions.append([])
        predictions[j] = tools.sigmoid_(np.dot(x, slopes[j]) + intercept[j])
    predictions = np.argmax(predictions, axis=0)
    print(predictions)
    house_predictions = [HOUSES[p] for p in predictions]
    print("tr", house_predictions)
    return np.mean(house_predictions == y)


def logistic_regression(
    x, y, cost_show=False, accuracy_show=False, stochastic=False, mini_batch=None
):
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
    if stochastic:
        mini_batch = 1
    slopes, intercept, cost_history = gradient_descent(
        x,
        y,
        slopes,
        intercept,
        compute_cost,
        compute_gradient,
        0,
        cost_show,
        mini_batch,
    )
    save_theta(slopes, intercept)
    if accuracy_show:
        print("Accuracy: ", compute_accuracy(x, y, slopes, intercept))
    if cost_show:
        show_cost(cost_history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", type=str, help="csv file")
    parser.add_argument(
        "-c", "--cost", action="store_true", help="Display and plot the cost function"
    )
    parser.add_argument(
        "-a",
        "--accuracy",
        action="store_true",
        help="Display the accuracy of the model",
    )
    parser.add_argument(
        "-s",
        "--stochastic",
        action="store_true",
        help="Use stochastic gradient descent",
    )
    parser.add_argument(
        "-mb", "--mini-batch", type=int, help="Use batch gradient descent"
    )
    args = parser.parse_args()
    try:
        tools.is_valid_path(args.csv_file)
    except Exception as e:
        print(e)
        sys.exit(e)

    x, y = tools.load_data(args.csv_file, "Hogwarts House")
    x = tools.normalize_df(x)
    logistic_regression(
        x, y, args.cost, args.accuracy, args.stochastic, args.mini_batch
    )
    # try:
    #     x, y = tools.load_data(
    #         args.csv_file, "Hogwarts House", transformHouses)
    #     logisticRegression(x, y)
    # except Exception as e:
    #     print(f"error {e.__class__.__name__}: {e}")
    #     sys.exit(e)
