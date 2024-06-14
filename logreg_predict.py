import ml_tools as tools
import sys
import pandas as pd
import argparse
import numpy as np

HOUSES = ["Slytherin", "Gryffindor", "Ravenclaw", "Hufflepuff"]


def predict(df, slopes, intercept):
    predictions = []
    for j in range(len(HOUSES)):
        predictions.append([])
        predictions[j] = tools.sigmoid_(np.dot(df, slopes[j]) + intercept[j])
    predictions = np.argmax(predictions, axis=0)
    print(predictions)
    house_predictions = [HOUSES[p] for p in predictions]
    df["Hogwarts House"] = house_predictions
    df.to_csv(
        "houses.csv",
        index=True,
        index_label="Index",
        header=True,
        columns=["Hogwarts House"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", type=str, help="csv file")
    parser.add_argument("slopes_path", type=str, help="csv file")
    parser.add_argument("intercept_path", type=str, help="csv file")
    args = parser.parse_args()
    try:
        tools.is_valid_path(args.csv_file)
        slopes = np.load(args.slopes_path)
        intercept = np.load(args.intercept_path)
    except Exception as e:
        sys.exit(e)
    df = (
        pd.read_csv(args.csv_file)
        .drop(["Index"], axis=1)
        .drop(["Hogwarts House"], axis=1)
        .select_dtypes(include=["float64", "int64"])
    )
    df.fillna(df.mean(), inplace=True)
    df = tools.normalize_df(df)
    predict(df, slopes, intercept)
