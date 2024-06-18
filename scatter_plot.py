import sys
import ml_tools as tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

HOUSE = "Hogwarts House"


def scatter_plot(df, feat1, feat2):
    tools.remove_empty_fields(df[feat1])
    tools.remove_empty_fields(df[feat2])
    plt.scatter(
        df[df[HOUSE] == "Slytherin"][feat1],
        df[df[HOUSE] == "Slytherin"][feat2],
        alpha=0.4,
        label="Slytherin",
        color="green",
    )
    plt.scatter(
        df[df[HOUSE] == "Gryffindor"][feat1],
        df[df[HOUSE] == "Gryffindor"][feat2],
        alpha=0.4,
        label="Gryffindor",
        color="red",
    )
    plt.scatter(
        df[df[HOUSE] == "Ravenclaw"][feat1],
        df[df[HOUSE] == "Ravenclaw"][feat2],
        alpha=0.4,
        label="Ravenclaw",
        color="cyan",
    )
    plt.scatter(
        df[df[HOUSE] == "Hufflepuff"][feat1],
        df[df[HOUSE] == "Hufflepuff"][feat2],
        alpha=0.4,
        label="Hufflepuff",
        color="gold",
    )
    plt.legend(loc="upper right")
    plt.xlabel(feat1)
    plt.ylabel(feat2)
    plt.title(feat1 + " vs " + feat2)
    plt.show()


if __name__ == "__main__":
    try:

        parser = argparse.ArgumentParser(description="Plot the pair_plot of a dataset")
        parser.add_argument("csv_file", type=str, help="csv file to plot")
        parser.add_argument("feat1", type=str, help="first column to plot")
        parser.add_argument("feat2", type=str, help="second column to plot")
        args = parser.parse_args()
        try:
            tools.is_valid_path(args.csv_file)
        except Exception as e:
            sys.exit(e)
        df = pd.read_csv(args.csv_file).drop(columns=["Index"])
        dropped = df.dropna(how="all", axis=1).select_dtypes([np.int64, np.float64])
        if args.feat1 not in dropped.columns or args.feat2 not in dropped.columns:
            sys.exit("Error: feature not valid")
        scatter_plot(df, args.feat1, args.feat2)
    except Exception as e:
        print(e)
        exit(1)
