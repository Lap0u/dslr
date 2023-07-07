import os
from os import path
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import math
import pandas as pd


def count(array):
    return len(array)


def load_data(filename: str):
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, :-1]  # Toutes les colonnes sauf la dernière
    y = data[:, -1]   # Dernière colonne
    return X, y


def mean(array):
    return sum(array) / len(array)


def min_(array):
    min = array[0]
    for i in range(len(array)):
        if (array[i] < min):
            min = array[i]
    return min


def quantile(array, q):
    return array[int(len(array) * q)]


def removeEmptyFields(array):
    x = [float(i) for i in array if not math.isnan(i)]
    return pd.DataFrame(x)


def max_(array):
    max = array[0]
    for i in range(len(array)):
        if (array[i] > max):
            max = array[i]
    return max


def std(array, mean):
    sum = 0
    for i in array:
        sum += (i - mean) ** 2
    return (sum / len(array)) ** 0.5


def isValidPath(filePath):
    if (path.isfile(filePath) == False):
        raise Exception('File does not exist')
    if (os.access(filePath, os.R_OK)) == False:
        raise Exception('File is not readable')
    if (Path(filePath).suffix != '.csv'):
        raise Exception('File is not a csv file')


def plot_data(X, y, pos_label="y=1", neg_label="y=0"):
    positive = y == 1
    negative = y == 0

    # Plot examples
    plt.plot(X[positive, 0], X[positive, 1], 'k+', label=pos_label)
    plt.plot(X[negative, 0], X[negative, 1], 'yo', label=neg_label)


def plot_decision_boundary(w, b, X, y):
    # Credit to dibgerge on Github for this plotting code

    plot_data(X[:, 0:2], y)

    if X.shape[1] <= 2:
        plot_x = np.array([min(X[:, 0]), max(X[:, 0])])
        plot_y = (-1. / w[1]) * (w[0] * plot_x + b)

        plt.plot(plot_x, plot_y, c="b")

    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((len(u), len(v)))

        # Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = sig(np.dot(map_feature(u[i], v[j]), w) + b)

        # important to transpose z before calling contour
        z = z.T

        # Plot z = 0.5
        plt.contour(u, v, z, levels=[0.5], colors="g")


def heatMap(df):
    sns.heatmap(df, annot=True)
    plt.show()


def normalize_array(X):
    return (X - X.min()) / (X.max() - X.min())


def sigmoid_(z):
    return 1 / (1 + np.exp(-z))
