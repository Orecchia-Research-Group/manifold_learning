### Collecting cells of interest

# In this model of psoriasis, tissue resident ILC2s are
# reprogramed to ILC3s. For analysis we are selecting for
# cells likely to be ILC2s or ILC3s on the basis of previously
# characterized populations.

import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
from ilc_loader import get_Cell_list, get_Gene_list, get_PCA_coord_var, get_Diff_coord_var

# Load in all cells in order to define norm_reads
ILC = sc.read("data/sct.h5ad")
#norm_reads = pd.DataFrame(ILC.layers["norm_data"].toarray())
norm_reads = ILC.layers["norm_data"].toarray()

# We have four labeled groups of cells that we want to collect together
ILC_var = sc.read("data/sct_variable.h5ad")
ILC2_ILC3 = ILC_var.obs["ilc2_ilc3"]
ILC3_Q = ILC_var.obs["quiescent_ilc3"]
ILC2_Q = ILC_var.obs["ilc2_quiescent"]
cloud_ILC3 = ILC_var.obs["cloud_ilc3"]

df = pd.DataFrame(dict(ILC2_ILC3 = ILC2_ILC3, ILC3_Q = ILC3_Q, ILC2_Q = ILC2_Q, cloud_ILC3 = cloud_ILC3))
cells = df.shape[0]

#binary yes ir no for if the cell is in our group
in_transition = np.zeros((cells, 1))

#Value of group - 0 if NA - over all cells
allcell_ILC2_ILC3 = np.zeros((cells, 1))
allcell_ILC3_Q = np.zeros((cells, 1))
allcell_ILC2_Q = np.zeros((cells, 1))
allcell_cloud_ILC3 = np.zeros((cells, 1))

#Cell labels - 0 if NA in all groups - over all cells
index_cells = np.zeros((cells, 1))
index_cells = index_cells.astype(str)

#Value of group - only cells in at least one group
val_ILC2_ILC3 = np.zeros((3807, 1))
val_ILC3_Q = np.zeros((3807, 1))
val_ILC2_Q = np.zeros((3807, 1))
val_cloud_ILC3 = np.zeros((3807, 1))

#Names of cells included in the value variables
index_val = np.array([])

#Variable for initially counting the number of cells in the groups of interest
number = 0

# Get Cell list for the following for loop
Cell_list = get_Cell_list()

#Loops through the different groups so that we collect all cells with a value in at least one of the groups
for i in range(0,cells):
	if np.isnan(df["ILC2_ILC3"][i]):
		if np.isnan(df["ILC3_Q"][i]):
			if np.isnan(df["ILC2_Q"][i]):
				if np.isnan(df["cloud_ILC3"][i]):
					in_transition[i,0] = 0
					allcell_ILC2_ILC3[i,0] = 0
					allcell_ILC3_Q[i,0] = 0
					allcell_ILC2_Q[i,0] = 0
					allcell_cloud_ILC3[i,0] = 0

					index_cells[i,0] = "0"

				else:
					in_transition[i,0] = 1

					allcell_ILC2_ILC3[i,0] = df["ILC2_ILC3"][i]
					allcell_ILC3_Q[i,0] = df["ILC3_Q"][i]
					allcell_ILC2_Q[i,0] = df["ILC2_Q"][i]
					allcell_cloud_ILC3[i,0] = df["cloud_ILC3"][i]

					index_cells[i,0] = Cell_list[i]

					val_ILC2_ILC3[number] = df["ILC2_ILC3"][i]
					val_ILC3_Q[number] = df["ILC3_Q"][i]
					val_ILC2_Q[number] = df["ILC2_Q"][i]
					val_cloud_ILC3[number] = df["cloud_ILC3"][i]

					index_val = np.append(index_val , df["ILC2_ILC3"].index[i])

					number += 1
			else:
				in_transition[i,0] = 1

				allcell_ILC2_ILC3[i,0] = df["ILC2_ILC3"][i]
				allcell_ILC3_Q[i,0] = df["ILC3_Q"][i]
				allcell_ILC2_Q[i,0] = df["ILC2_Q"][i]
				allcell_cloud_ILC3[i,0] = df["cloud_ILC3"][i]

				index_cells[i,0] = Cell_list[i]

				val_ILC2_ILC3[number] = df["ILC2_ILC3"][i]
				val_ILC3_Q[number] = df["ILC3_Q"][i]
				val_ILC2_Q[number] = df["ILC2_Q"][i]
				val_cloud_ILC3[number] = df["cloud_ILC3"][i]

				index_val = np.append(index_val , df["ILC2_ILC3"].index[i])

				number += 1

		else:
			in_transition[i,0] = 1

			allcell_ILC2_ILC3[i,0] = df["ILC2_ILC3"][i]
			allcell_ILC3_Q[i,0] = df["ILC3_Q"][i]
			allcell_ILC2_Q[i,0] = df["ILC2_Q"][i]
			allcell_cloud_ILC3[i,0] = df["cloud_ILC3"][i]

			index_cells[i,0] = Cell_list[i]

			val_ILC2_ILC3[number] = df["ILC2_ILC3"][i]
			val_ILC3_Q[number] = df["ILC3_Q"][i]
			val_ILC2_Q[number] = df["ILC2_Q"][i]
			val_cloud_ILC3[number] = df["cloud_ILC3"][i]

			index_val = np.append(index_val , df["ILC2_ILC3"].index[i])

			number += 1
	else:
		in_transition[i,0] = 1

		allcell_ILC2_ILC3[i,0] = df["ILC2_ILC3"][i]
		allcell_ILC3_Q[i,0] = df["ILC3_Q"][i]
		allcell_ILC2_Q[i,0] = df["ILC2_Q"][i]
		allcell_cloud_ILC3[i,0] = df["cloud_ILC3"][i]

		index_cells[i,0] = Cell_list[i]

		val_ILC2_ILC3[number] = df["ILC2_ILC3"][i]
		val_ILC3_Q[number] = df["ILC3_Q"][i]
		val_ILC2_Q[number] = df["ILC2_Q"][i]
		val_cloud_ILC3[number] = df["cloud_ILC3"][i]

		index_val = np.append(index_val , df["ILC2_ILC3"].index[i])

		number += 1

print(number)

#There are some nan values in these arrays that we want to force to zero
val_ILC2_ILC3[np.isnan(val_ILC2_ILC3)] = 0
val_ILC3_Q[np.isnan(val_ILC3_Q)] = 0
val_ILC2_Q[np.isnan(val_ILC2_Q)] = 0
val_cloud_ILC3[np.isnan(val_cloud_ILC3)] = 0

### Creating normalized read matrix for only the cells of interest
#Numpy array with normalized reads for all cells
#norm_reads = np.array(norm_reads)
#PCA_coord = np.array(PCA_coord)
#Diff_coord = np.array(Diff_coord)
PCA_coord = get_PCA_coord_var()
Diff_coord = get_Diff_coord_var()

#Number of genes to initialize new matrix
genes = get_Gene_list().shape[0]
reads = np.zeros((1,15806 ))

PCA_reads = np.zeros((1,50))
Diff_reads = np.zeros((1,15))

#Going through all of the cells, if the  associated index in index_cell is nonzero then
#it was included in our group and we should pull the associated gene expression data
print("Going through all of the cells.")
print("If the  associated index in index_cell is nonzero then it was included in our group,")
print("and we should pull the associated gene expression data")
for i in tqdm(range(cells)):
	j = index_cells[i]
	if j != "0":
		next_cell = norm_reads[i,:]
		next_cell = np.reshape(next_cell, (1,15806 ))
		reads = np.append(reads, next_cell, axis = 0)

		next_PCA = PCA_coord[i,:]
		next_PCA = np.reshape(next_PCA, (1,50))
		PCA_reads = np.append(PCA_reads, next_PCA, axis = 0)

		next_Diff = Diff_coord[i,:]
		next_Diff = np.reshape(next_Diff, (1,15))
		Diff_reads = np.append(Diff_reads, next_Diff, axis = 0)

# To get the indices to work out I have a row of zeros at the top that I need to omit
reads = reads[1:, :]
PCA_reads = PCA_reads[1:, :]
Diff_reads = Diff_reads[1:, :]

### Save matrix with cells of interest
np.savetxt("data/ILCs_reads.csv", reads, delimiter=',')
np.savetxt("data/ILCs_index.csv", index_val, delimiter=',', fmt="%s")
np.savetxt("data/index_cells.csv", index_cells, delimiter=',', fmt="%s")
