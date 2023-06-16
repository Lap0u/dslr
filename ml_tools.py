import os
from os import path
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

def isValidPath(filePath):
  if (path.isfile(filePath) == False):
    raise Exception('File does not exist')
  if (os.access(filePath, os.R_OK)) == False:
    raise Exception('File is not readable')
  if (Path(filePath).suffix != '.csv'):
    raise Exception('File is not a csv file')

def heatMap(df):
  sns.heatmap(df, annot=True)
  plt.show()

def normalize_array(X):
  return (X - X.min()) / (X.max() - X.min())

def sigmoid_(x):
  return 1 / (1 + np.exp(-x))