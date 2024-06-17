import os
from os import path
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import math
import pandas as pd
from sklearn.metrics import confusion_matrix


def count(array):
    return len(array)


def get_mini_batches(x, transformed_y, batch):
    rand_val = np.random.randint(0, len(x), batch)
    batched_x = x.iloc[rand_val]
    batched_y = transformed_y[rand_val]
    return batched_x, batched_y


def load_data(filename: str, target_col_name: str):
    df = pd.read_csv(filename, sep=",").drop(["Index"], axis=1)
    y = df["Hogwarts House"]
    df = df.select_dtypes(include=["float64", "int64"])
    df.fillna(df.mean(), inplace=True)
    return df, y


def mean(series: pd.Series):
    return series[0].sum() / len(series)


def min_(array):
    min_ = array[0]
    for i in range(len(array)):
        if array[i] < min_:
            min_ = array[i]
    return min_


def quantile(array, q):
    return array[0][int(len(array) * q)]


def remove_empty_fields(array):
    x = [float(i) for i in array if not math.isnan(i)]
    return pd.DataFrame(x)


def max_(array):
    max_ = array[0]
    for i in range(len(array)):
        if array[i] > max_:
            max_ = array[i]
    return max_


def std(array, mean):
    sum_ = 0
    for i in array[0]:
        sum_ += (i - mean) ** 2
    return (sum_ / (len(array[0]) - 1)) ** 0.5


def is_valid_path(file_path):
    if path.isfile(file_path) is False:
        raise Exception("File does not exist")
    if (os.access(file_path, os.R_OK)) is False:
        raise Exception("File is not readable")
    if Path(file_path).suffix != ".csv":
        raise Exception("File is not a csv file")


def plot_data(x, y, pos_label="y=1", neg_label="y=0"):
    positive = y == 1
    negative = y == 0

    # Plot examples
    plt.plot(x[positive, 0], x[positive, 1], "k+", label=pos_label)
    plt.plot(x[negative, 0], x[negative, 1], "yo", label=neg_label)


def plot_confusion_matrix(houses, house_predictions, labels):
    cm = confusion_matrix(houses, house_predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="coolwarm", xticklabels=labels, yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


def map_feature(x1, x2):
    """
    Feature mapping function to polynomial features
    """
    x1 = np.atleast_1d(x1)
    x2 = np.atleast_1d(x2)
    degree = 6
    out = []
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out.append((x1 ** (i - j) * (x2**j)))
    return np.stack(out, axis=1)


def plot_decision_boundary(w, b, x, y):
    # Credit to dibgerge on Github for this plotting code
    print("w", w)
    print("b", b)
    print("x", x)
    print("y", y)
    plot_data(x[:, 0:2], y)

    if x.shape[1] <= 2:
        plot_x = np.array([min(x[:, 0]), max(x[:, 0])])
        plot_y = (-1.0 / w[1]) * (w[0] * plot_x + b)

        plt.plot(plot_x, plot_y, c="b")

    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((len(u), len(v)))

        # Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = sigmoid_(np.dot(map_feature(u[i], v[j]), w) + b)

        # important to transpose z before calling contour
        z = z.T

        # Plot z = 0.5
        plt.contour(u, v, z, levels=[0.5], colors="g")


def mad(array, mean):
    sum_ = 0
    for i in array[0]:
        sum_ += abs(i - mean)
    return sum_ / len(array[0])


def skew(array, mean):
    sum_ = 0
    for i in array[0]:
        sum_ += (i - mean) ** 3
    return sum_ / len(array[0]) / (std(array, mean) ** 3)


def heatmap(df):
    plt.figure(figsize=(12, 8))
    plt.title("Correlation Heatmap")
    plt.margins(1)
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.show()


def normalize_array(x):
    return (x - x.min()) / (x.max() - x.min())


def normalize_df(df):
    for column in df.columns:
        if df[column].dtype != "object":
            df[column] = normalize_array(df[column])
    return df


def denormalize_array(list, elem):
    return (elem * (max(list) - min(list))) + min(list)


def sigmoid_(z):
    return 1 / (1 + np.exp(-z))
