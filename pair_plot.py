import sys
import numpy as np
import pandas as pd
import ml_tools as tools
import matplotlib.pyplot as plt


def histogram(df, col, length, i, j):
    print('pos hist', (i-1) * length + j)
    plt.subplot(length + 1, length + 1, (i - 1) * length + j)
    plt.hist(df[df['Hogwarts House'] == 'Slytherin'][col],
             alpha=0.4, label="Slytherin", color="green")
    plt.hist(df[df['Hogwarts House'] == 'Gryffindor'][col],
             alpha=0.4, label="Gryffindor", color="red")
    plt.hist(df[df['Hogwarts House'] == 'Ravenclaw'][col],
             alpha=0.4, label="Ravenclaw", color="cyan")
    plt.hist(df[df['Hogwarts House'] == 'Hufflepuff'][col],
             alpha=0.4, label="Hufflepuff", color="gold")


def scatter_plot(df, row, col, length, i, j):
    print('pos scat', (i-1) * length + j)
    plt.subplot(length, length + 1, (i - 1) * length + j)
    plt.scatter(df[df['Hogwarts House'] == 'Slytherin'][row], df[df['Hogwarts House']
                == 'Slytherin'][col], alpha=0.4, s=1, label="Slytherin", color="green")
    plt.scatter(df[df['Hogwarts House'] == 'Gryffindor'][row], df[df['Hogwarts House']
                == 'Gryffindor'][col], alpha=0.4, s=1, label="Gryffindor", color="red")
    plt.scatter(df[df['Hogwarts House'] == 'Ravenclaw'][row], df[df['Hogwarts House']
                == 'Ravenclaw'][col], alpha=0.4, s=1, label="Ravenclaw", color="cyan")
    plt.scatter(df[df['Hogwarts House'] == 'Hufflepuff'][row], df[df['Hogwarts House']
                == 'Hufflepuff'][col], alpha=0.4, s=1, label="Hufflepuff", color="gold")


def pair_plot(df):
    dropped = df.dropna(how='all', axis=1).select_dtypes(
        [np.int64, np.float64])
    length = len(dropped.columns)
    plt.rcParams["figure.figsize"] = [50, 42]
    for row, i in zip(dropped.columns, range(1, length + 1)):
        for col, j in zip(dropped.columns, range(1, length + 1)):
            print(i, j, row, col)
            tools.removeEmptyFields(df[col])
            tools.removeEmptyFields(df[row])
            if row == col:
                histogram(df, row, length, i, j)
            else:
                scatter_plot(df, row, col, length, i, j)
    plt.legend(loc='upper right')
    plt.title("Pair plot")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        sys.exit("Usage: python3 describe.py <csv file>")
    try:
        tools.isValidPath(sys.argv[1])
    except Exception as e:
        sys.exit(e)
    df = pd.read_csv(sys.argv[1]).drop(columns=['Index'])
    pair_plot(df)
