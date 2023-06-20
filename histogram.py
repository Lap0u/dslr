import sys
import pandas as pd
import matplotlib.pyplot as plt
from ml_tools import isValidPath
from ml_tools import normalize_array
from ml_tools import removeEmptyFields

def histogram(df):
  for column in df.columns:
    if df[column].dtype != "object":
      cleaned = removeEmptyFields(df[column])
      if len(cleaned) == 0:
        continue
      df[column] = normalize_array(cleaned)
      print("Standard deviation of " + column + " : " + str(df[column].std()))
      plt.hist(df[df['Hogwarts House'] == 'Slytherin'][column], alpha=0.4, label="Slytherin", color="green")
      plt.hist(df[df['Hogwarts House'] == 'Gryffindor'][column], alpha=0.4, label="Gryffindor", color="red")
      plt.hist(df[df['Hogwarts House'] == 'Ravenclaw'][column], alpha=0.4, label="Ravenclaw", color="cyan")
      plt.hist(df[df['Hogwarts House'] == 'Hufflepuff'][column], alpha=0.4, label="Hufflepuff", color="gold")
      plt.xlabel(column)
      plt.ylabel("Frequency")
      plt.legend(loc='upper right')
      plt.title("Histogram of " + column)
      plt.show()

if __name__ == "__main__":
	if len(sys.argv) <= 1:
		sys.exit("Usage: python3 train.py <csv file>")
	try:
		isValidPath(sys.argv[1])
	except Exception as e:
		sys.exit(e)
	df = pd.read_csv(sys.argv[1]).drop(columns=['Index'])
	histogram(df)