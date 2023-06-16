import sys
import numpy as np
import pandas as pd
from ml_tools import isValidPath

def createIndexedDf(column):
	print('col', column)
	return pd.DataFrame(columns=column, index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"])

def compute(df, described):
	for column in df:
		if df[column].dtype != "object":
			described[column]["count"] = df[column].count()
			described[column]["mean"] = df[column].mean()
			described[column]["std"] = df[column].std()
			described[column]["min"] = df[column].min()
			described[column]["25%"] = df[column].quantile(.25)
			described[column]["50%"] = df[column].quantile(.5)
			described[column]["75%"] = df[column].quantile(.75)
			described[column]["max"] = df[column].max()

def describe(df):
	described = createIndexedDf(df.select_dtypes([np.int64,np.float64]).columns)
	compute(df, described)
	print(described)


if __name__ == "__main__":
	if len(sys.argv) <= 1:
		sys.exit("Usage: python3 train.py <csv file>")
	try:
		isValidPath(sys.argv[1])
	except Exception as e:
		sys.exit(e)
	df = pd.read_csv(sys.argv[1])
	describe(df)
