import sys
import matplotlib.pyplot as plt
import numpy as np
import ml_tools as tools
import argparse

ALPHA = 0.01
EPOCHS = 5000
EPSILON = 1e-15
HOUSES = ["Slytherin", "Gryffindor", "Ravenclaw", "Hufflepuff"]


def compute_cost(x, y, slopes, intercept):
    """
    Computes the cost over all examples
    Args:
        x : (ndarray Shape (m,n)) data, m examples by n features
        y : (ndarray Shape (m,))  target value
        slopes : (ndarray Shape (n,))  values of parameters of the model
        intercept : (scalar)              value of bias parameter of the model
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


def compute_gradient(x, y, slopes, intercept):
    """
    Computes the gradient for logistic regression

    Args:
        x : (ndarray Shape (m,n)) data, m examples by n features
        y : (ndarray Shape (m,))  target value
        slopes : (ndarray Shape (n,))  values of parameters of the model
        intercept : (scalar)              value of bias parameter of the model
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
    cost_show,
    batch=None,
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
    if cost_show:
        colors = ["red", "blue", "green", "magenta"]
        fig, axs = plt.subplots(2, 2)
        fig.suptitle("Cost function")
        fig.tight_layout(pad=3.0)

    for j in range(len(HOUSES)):
        cost_history.append([])
        intercept.append([])
        slopes.append([])
        intercept[j] = intercept_s
        slopes[j] = slopes_s
        transformed_y = transform_houses(y, HOUSES[j])
        for i in range(EPOCHS):
            batched_x, batched_y = x, transformed_y
            if batch:
                batched_x, batched_y = tools.get_mini_batches(x, transformed_y, batch)
            if cost_show:
                cost = cost_function(batched_x, batched_y, slopes[j], intercept[j])
                if i % 100 == 0:
                    print(f"Epoch: {i} <-> Cost: {cost}")
                cost_history[j].append(cost)
                show_cost(cost_history, colors, axs, i, j)

            dj_dintercept, dj_dslopes = gradient_function(
                batched_x, batched_y, slopes[j], intercept[j]
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


def show_cost(cost_history, colors, axs, epoch, curr_cost_index):
    if epoch % 1000 == 0:
        axs[curr_cost_index // 2, curr_cost_index % 2].plot(
            cost_history[curr_cost_index], color=colors[curr_cost_index]
        )
        axs[curr_cost_index // 2, curr_cost_index % 2].set_title(
            f"{HOUSES[curr_cost_index]} vs all"
        )
        plt.pause(0.001)


def compute_accuracy(x, y, slopes, intercept):
    predictions = []
    for j in range(len(HOUSES)):
        predictions.append([])
        predictions[j] = tools.sigmoid_(np.dot(x, slopes[j]) + intercept[j])
    predictions = np.argmax(predictions, axis=0)
    house_predictions = [HOUSES[p] for p in predictions]
    return np.mean(house_predictions == y)


def compute_confusion_matrix(x, y, slopes, intercept):
    predictions = []
    for j in range(len(HOUSES)):
        predictions.append([])
        predictions[j] = tools.sigmoid_(np.dot(x, slopes[j]) + intercept[j])
    predictions = np.argmax(predictions, axis=0)
    house_predictions = [HOUSES[p] for p in predictions]
    tools.plot_confusion_matrix(y, house_predictions, HOUSES)


def logistic_regression(
    x,
    y,
    cost_show=False,
    accuracy_show=False,
    stochastic=False,
    mini_batch=None,
    confusion_matrix=False,
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
    slopes, intercept, _ = gradient_descent(
        x,
        y,
        slopes,
        intercept,
        compute_cost,
        compute_gradient,
        cost_show,
        mini_batch,
    )
    save_theta(slopes, intercept)
    if accuracy_show:
        print("Accuracy: ", compute_accuracy(x, y, slopes, intercept))
    if confusion_matrix:
        compute_confusion_matrix(x, y, slopes, intercept)


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
        "-cm",
        "--confusion-matrix",
        action="store_true",
        help="Display the confusion matrix",
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
        x,
        y,
        args.cost,
        args.accuracy,
        args.stochastic,
        args.mini_batch,
        args.confusion_matrix,
    )
