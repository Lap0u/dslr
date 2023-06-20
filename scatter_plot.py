import sys
import ml_tools as tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def scatter_plot(df, feat1, feat2):
	tools.removeEmptyFields(df[feat1])
	tools.removeEmptyFields(df[feat2])
	plt.scatter(df[df['Hogwarts House'] == 'Slytherin'][feat1], df[df['Hogwarts House'] == 'Slytherin'][feat2], alpha=0.4, label="Slytherin", color="green")
	plt.scatter(df[df['Hogwarts House'] == 'Gryffindor'][feat1], df[df['Hogwarts House'] == 'Gryffindor'][feat2], alpha=0.4, label="Gryffindor", color="red")
	plt.scatter(df[df['Hogwarts House'] == 'Ravenclaw'][feat1], df[df['Hogwarts House'] == 'Ravenclaw'][feat2], alpha=0.4, label="Ravenclaw", color="cyan")
	plt.scatter(df[df['Hogwarts House'] == 'Hufflepuff'][feat1], df[df['Hogwarts House'] == 'Hufflepuff'][feat2], alpha=0.4, label="Hufflepuff", color="gold")
	plt.legend(loc='upper right')
	plt.xlabel(feat1)
	plt.ylabel(feat2)
	plt.title(feat1 + " vs " + feat2)
	plt.show()

if __name__ == "__main__":
	if len(sys.argv) != 4:
		sys.exit("Usage: python3 train.py <csv file> <feature 1> <feature 2>")
	try:
		tools.isValidPath(sys.argv[1])
	except Exception as e:
		sys.exit(e)
	feat1 = sys.argv[2]
	feat2 = sys.argv[3]
	df = pd.read_csv(sys.argv[1]).drop(columns=['Index'])
	dropped = df.dropna(how='all', axis=1).select_dtypes([np.int64,np.float64])
	if not feat1 in dropped.columns or not feat2 in dropped.columns:
		sys.exit("Error: feature not valid")
	scatter_plot(df, feat1, feat2)