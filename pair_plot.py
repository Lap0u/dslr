import sys
import numpy as np
import pandas as pd
import ml_tools as tools
import matplotlib.pyplot as plt
import argparse

HOUSE = "Hogwarts House"


def histogram(df, col, length, i, j):
    plt.subplot(length, length, (i - 1) * length + j)
    plt.hist(
        df[df[HOUSE] == "Slytherin"][col], alpha=0.4, label="Slytherin", color="green"
    )
    plt.hist(
        df[df[HOUSE] == "Gryffindor"][col], alpha=0.4, label="Gryffindor", color="red"
    )
    plt.hist(
        df[df[HOUSE] == "Ravenclaw"][col], alpha=0.4, label="Ravenclaw", color="cyan"
    )
    plt.hist(
        df[df[HOUSE] == "Hufflepuff"][col], alpha=0.4, label="Hufflepuff", color="gold"
    )
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    if j == 1:
        ax.set_ylabel(col.replace(" ", "\n"), fontsize=6)
    if i == length:
        ax.set_xlabel(col.replace(" ", "\n"), fontsize=8)


def scatter_plot(df, row, col, length, i, j):
    plt.subplot(length, length, (i - 1) * length + j)
    plt.scatter(
        df[df[HOUSE] == "Slytherin"][row],
        df[df[HOUSE] == "Slytherin"][col],
        alpha=0.4,
        s=0.1,
        label="Slytherin",
        color="green",
    )
    plt.scatter(
        df[df[HOUSE] == "Gryffindor"][row],
        df[df[HOUSE] == "Gryffindor"][col],
        alpha=0.4,
        s=0.4,
        label="Gryffindor",
        color="red",
    )
    plt.scatter(
        df[df[HOUSE] == "Ravenclaw"][row],
        df[df[HOUSE] == "Ravenclaw"][col],
        alpha=0.4,
        s=0.4,
        label="Ravenclaw",
        color="cyan",
    )
    plt.scatter(
        df[df[HOUSE] == "Hufflepuff"][row],
        df[df[HOUSE] == "Hufflepuff"][col],
        alpha=0.4,
        s=0.4,
        label="Hufflepuff",
        color="gold",
    )
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    if j == 1:
        ax.set_ylabel(row.replace(" ", "\n"), fontsize=6)
    if i == length:
        ax.set_xlabel(col.replace(" ", "\n"), fontsize=8)


def pair_plot(df, heatmap=False):
    dropped = df.dropna(how="all", axis=1).select_dtypes([np.int64, np.float64])
    length = len(dropped.columns)
    plt.rcParams["figure.figsize"] = [50, 42]
    for row, i in zip(dropped.columns, range(1, length + 1)):
        for col, j in zip(dropped.columns, range(1, length + 1)):
            tools.remove_empty_fields(df[col])
            tools.remove_empty_fields(df[row])
            if row == col:
                histogram(df, row, length, i, j)
            else:
                scatter_plot(df, row, col, length, i, j)
    plt.suptitle("Pair plot for each combination of two features", fontsize=20)
    plt.legend(bbox_to_anchor=(1.04, 1))
    plt.show()
    if args.heatmap:
        tools.heatmap(dropped)


if __name__ == "__main__":
    try:

        parser = argparse.ArgumentParser(description="Plot the pair_plot of a dataset")
        parser.add_argument("csv_file", type=str, help="csv file to plot")
        parser.add_argument(
            "-hm", "--heatmap", action="store_true", help="plot heatmap"
        )
        args = parser.parse_args()
        try:
            tools.is_valid_path(args.csv_file)
        except Exception as e:
            sys.exit(e)
        df = pd.read_csv(args.csv_file).drop(columns=["Index"])
        pair_plot(df, args.heatmap)
    except Exception as e:
        print(e)
        exit(1)
