import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ml_tools import isValidPath

def train_model(filePath):
  data = pd.read_csv(filePath)
  print(data)

if __name__ == '__main__':
	if (len(sys.argv) <= 1):
		sys.exit('Usage: python3 train.py <csv file>')
	try:
		isValidPath(sys.argv[1])
	except Exception as e:
		sys.exit(e)
	train_model(sys.argv[1])
