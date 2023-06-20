import sys
import math
import numpy as np
import pandas as pd
from ml_tools import isValidPath
from ml_tools import count
from ml_tools import mean
from ml_tools import std
from ml_tools import quantile

def createIndexedDf(column):
	return pd.DataFrame(columns=column, index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"])

def compute(df, described):
	for column in described.columns:
		if df[column].dtype != "object":
			sorted = df[column].sort_values()
			new_data = [float(i) for i in sorted if not math.isnan(i)]
			print('new', new_data)
			described[column]["count"] = count(new_data)
			described[column]["mean"] = mean(new_data)
			described[column]["std"] = std(new_data, described[column]["mean"])
			described[column]["min"] = new_data[0]
			described[column]["25%"] = quantile(new_data, .25)
			described[column]["50%"] = quantile(new_data, .5)
			described[column]["75%"] = quantile(new_data, .75)
			described[column]["max"] = new_data[len(new_data) - 1]
	return described

def describe(df):
	struct = createIndexedDf(df.dropna(how='all', axis=1).select_dtypes([np.int64,np.float64]).columns)
	described = compute(df, struct)
	print (described)

if __name__ == "__main__":
	if len(sys.argv) <= 1:
		sys.exit("Usage: python3 train.py <csv file>")
	try:
		isValidPath(sys.argv[1])
	except Exception as e:
		sys.exit(e)
	df = pd.read_csv(sys.argv[1])
	describe(df)
