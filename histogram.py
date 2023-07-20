import sys
import pandas as pd
import matplotlib.pyplot as plt
import ml_tools as tools
import argparse

HOUSE = 'Hogwarts House'


def histogram(df):
    for column in df.columns:
        if df[column].dtype != "object":
            cleaned = tools.remove_empty_fields(df[column])
            if len(cleaned) == 0:
                continue
            df[column] = tools.normalize_array(cleaned)
            print("Standard deviation of " + column +
                  " : " + str(df[column].std()))
            plt.hist(df[df[HOUSE] == 'Slytherin'][column],
                     alpha=0.4, label="Slytherin", color="green")
            plt.hist(df[df[HOUSE] == 'Gryffindor']
                     [column], alpha=0.4, label="Gryffindor", color="red")
            plt.hist(df[df[HOUSE] == 'Ravenclaw'][column],
                     alpha=0.4, label="Ravenclaw", color="cyan")
            plt.hist(df[df[HOUSE] == 'Hufflepuff']
                     [column], alpha=0.4, label="Hufflepuff", color="gold")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.legend(loc='upper right')
            plt.title("Histogram of " + column)
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot all histograms of a dataset")
    parser.add_argument("csv_file", type=str, help="csv file to plot")
    args = parser.parse_args()
    try:
        tools.is_valid_path(args.csv_file)
    except Exception as e:
        sys.exit(e)
    df = pd.read_csv(args.csv_file).drop(columns=['Index'])
    histogram(df)
