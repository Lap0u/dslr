import ml_tools as tools
import sys
import pandas as pd
import argparse


def predict(df):
    print(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file', type=str, help='csv file')
    parser.add_argument('weight', type=str, help='csv file')
    args = parser.parse_args()
    try:
        tools.isValidPath(args.csv_file)
        tools.isValidPath(args.weight)
    except Exception as e:
        sys.exit(e)
    df = pd.read_csv(args.csv_file)
    predict(df)
