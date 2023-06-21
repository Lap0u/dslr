import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ml_tools import isValidPath
from ml_tools import heatMap

def logisticRegression(df):
  print(df)

def basicLogisticRegression(df):
  print(df)
  heatMap(df)
  # heatMap(df)
  # print(df)

if __name__ == '__main__':
	if (len(sys.argv) <= 1):
		sys.exit('Usage: python3 train_model.py <csv file>')
	try:
		isValidPath(sys.argv[1])
	except Exception as e:
		sys.exit(e)
	df = pd.read_csv(sys.argv[1])
	print(df.head())
	basicLogisticRegression(df)
	# logisticRegression(df)
