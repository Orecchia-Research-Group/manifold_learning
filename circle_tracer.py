# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd
import scanpy as sc
from ripser import Rips
from ripser import ripser
from time import time
import pickle

ILC_var = sc.read("data/sct_variable.h5ad")

try:
	Cell_list = pd.read_csv("data/hieromnimon/Cell_list_ILC.csv", sep=",", header=0)
except FileNotFoundError:
	Obs = ILC_var.obs
	O = Obs["ilc2_ilc3"].to_frame()
	O.to_csv("data/hieromnimon/Cell_list_ILC.csv", sep=",")
	Cell_list = pd.read_csv("data/hieromnimon/Cell_list_ILC.csv", sep=",", header=0)
Cell_list = np.array(Cell_list)
Cell_list = Cell_list[:, 0]
try:
	Gene_list = pd.read_csv("data/hieromnimon/Gene_list_ILC.csv", sep=",", header=0)
except FileNotFoundError:
	ILC_var.var["Selected"].to_csv("data/hieromnimon/Gene_list_ILC.csv", sep=",")
	Gene_list = pd.read_csv("data/hieromnimon/Gene_list_ILC.csv", sep=",", header=0)
Gene_list = np.array(Gene_list)
Gene_list = Gene_list[:, 0]

matrix = ILC_var.layers["norm_data"]
matrix = np.array(matrix.toarray())

ILC_total_reads = pd.DataFrame(matrix, index=Cell_list, columns=Gene_list)
np_data = np.array(ILC_total_reads)

# We have four labeled groups of cells that we want to collect together
ILC2_ILC3 = ILC_var.obs["ilc2_ilc3"]
ILC3_Q = ILC_var.obs["quiescent_ilc3"]
ILC2_Q = ILC_var.obs["ilc2_quiescent"]
cloud_ILC3 = ILC_var.obs["cloud_ilc3"]

df = pd.DataFrame(dict(ILC2_ILC3=ILC2_ILC3, ILC3_Q=ILC3_Q, ILC2_Q=ILC2_Q, cloud_ILC3=cloud_ILC3))

pd_data = pd.DataFrame(np_data, index=df.index)

subset = []
for label in df.index:
	if not np.isnan(df.loc[label]["ILC2_ILC3"]):
		subset.append(label)
	elif not np.isnan(df.loc[label]["ILC2_Q"]):
		subset.append(label)
	elif not np.isnan(df.loc[label]["ILC3_Q"]):
		subset.append(label)
	elif not np.isnan(df.loc[label]["cloud_ILC3"]):
		subset.append(label)

topic_cells = np.array(pd_data.loc[subset])

# Do ripster shenanigans
start = time()
rips = Rips()
diagrams = rips.fit_transform(np_data, distance_matrix=False, metric="euclidean")
print("Time elapsed: "+str(time() - start))

with open("ripser_output.pkl", "wb") as file:
	pickle.dump(diagrams, file)
