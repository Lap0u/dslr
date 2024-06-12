import sys
import numpy as np
import pandas as pd
import argparse
import ml_tools as tools


def create_indexed_df(column):
    return pd.DataFrame(
        columns=column,
        index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
    )


def compute(df, described):
    for column in described.columns:
        if df[column].dtype != "object":
            sorted_values = df[column].sort_values()
            cleaned = tools.remove_empty_fields(sorted_values)
            described[column]["count"] = tools.count(cleaned)
            described[column]["mean"] = tools.mean(cleaned)
            described[column]["std"] = tools.std(cleaned, described[column]["mean"])
            described[column]["min"] = cleaned[0][0]
            described[column]["25%"] = tools.quantile(cleaned, 0.25)
            described[column]["50%"] = tools.quantile(cleaned, 0.5)
            described[column]["75%"] = tools.quantile(cleaned, 0.75)
            described[column]["max"] = cleaned[0][len(cleaned[0]) - 1]
    return described


def describe(df):
    struct = create_indexed_df(
        df.dropna(how="all", axis=1).select_dtypes([np.int64, np.float64]).columns
    )
    described = compute(df, struct)
    print(described)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Describe a dataset")
    parser.add_argument("csv_file", type=str, help="csv file to describe")
    parser.add_argument(
        "-n", "--normalize", action="store_true", help="normalize column"
    )
    args = parser.parse_args()
    try:
        tools.is_valid_path(args.csv_file)
    except Exception as e:
        sys.exit(e)
    df = pd.read_csv(args.csv_file)
    print(df.describe())  # real one
    if args.normalize:
        df = tools.normalize_df(df)
    describe(df)
