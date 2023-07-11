import sys
import pandas as pd
import matplotlib.pyplot as plt
import ml_tools as tools

HOUSE = 'Hogwarts House'


def histogram(df):
    for column in df.columns:
        if df[column].dtype != "object":
            cleaned = tools.removeEmptyFields(df[column])
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
    if len(sys.argv) <= 1:
        sys.exit("Usage: python3 histogram.py <csv file>")
    try:
        tools.isValidPath(sys.argv[1])
    except Exception as e:
        sys.exit(e)
    df = pd.read_csv(sys.argv[1]).drop(columns=['Index'])
    histogram(df)
