import os
from os import path
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import math
import pandas as pd

def count(array):
  return len(array)

def mean(array):
  return sum(array) / len(array)

def min_(array):
  min = array[0]
  for i in range(len(array)):
    if (array[i] < min):
      min = array[i]
  return min

def quantile(array, q):
  return array[int(len(array) * q)]

def removeEmptyFields(array):
  x = [float(i) for i in array if not math.isnan(i)]
  return pd.DataFrame(x)

def max_(array):
  max = array[0]
  for i in range(len(array)):
    if (array[i] > max):
      max = array[i]
  return max

def std(array, mean):
  sum = 0
  for i in array:
    sum += (i - mean) ** 2
  return (sum / len(array)) ** 0.5

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