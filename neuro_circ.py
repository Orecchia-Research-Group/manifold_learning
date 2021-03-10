import numpy as np
import pandas as pd

df_1 = pd.read_csv("data/neuro_circ/sheet_1.csv", header=None).T
df_2 = pd.read_csv("data/neuro_circ/sheet_2.csv", header=None).T

# NOTE: within each of {df_1, df_2}, Cell_number is a unique ID,
# so each row (post-transpose) is a unique cell

# Set columns
df_1.columns = df_1.iloc[0]
df_1.drop(df_1.index[0], inplace=True)
df_2.columns = df_2.iloc[0]
df_2.drop(df_2.index[0], inplace=True)

names_1 = df_1["Animal_Number"].tolist()
names_2 = df_2["Animal_Number"].tolist()
names = names_1 + names_2
uniq_names = np.unique(names)

groups_1 = df_1["core_group"].tolist()
groups_2 = df_2["core_group"].tolist()
groups = groups_1 + groups_2
uniq_groups = np.unique(groups)

# Drop irrelevant columns
to_drop = ['Cell_number', 'Animal_Number', 'core_group',
	'Light_treat', 'Rostral_Caudal', 
	'Ventral_Dorsal', 'Medial_Lateral']
df_1.drop(labels=to_drop, axis=1, inplace=True)
df_2.drop(labels=to_drop, axis=1, inplace=True)

