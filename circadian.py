import numpy as np
import pandas as pd
import pickle

df = pd.read_csv("data/circadian/matrix/series_matrix.txt",
	sep=None, engine="python", index_col="ID_REF")

row_names = df.index.tolist()

for name in row_names:
	print(name)
