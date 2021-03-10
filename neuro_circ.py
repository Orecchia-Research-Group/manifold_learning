import numpy as np
import pandas as pd

df_1 = pd.read_csv("data/neuro_circ/sheet_1.csv").T
df_2 = pd.read_csv("data/neuro_circ/sheet_2.csv").T

print(df_1.index)
