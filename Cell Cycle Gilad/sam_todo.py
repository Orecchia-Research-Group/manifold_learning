import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ripser import Rips, ripser
from persim import plot_diagrams
import bats
from tqdm import tqdm
from time import time

seed = 42
np.random.seed(seed)

# Read in data
gene_expression = pd.read_csv("final_geneexpression.csv", sep=',', header = 0, index_col = 0)
labels = pd.read_csv("final_labels.csv", sep=',', header = 0, index_col = 0)

# Get column names for key genes in cell cycles
CDK1 = "ENSG00000170312"
UBE2C = "ENSG00000175063"
TOP2A = "ENSG00000131747"
#H4C5 = "ENSG00000276966"
H4C3 = "ENSG00000197061"

cell_cycle = [CDK1, UBE2C, TOP2A, H4C3]
compare = gene_expression.loc[cell_cycle]

# Bin expression - find bins for cell cycle genes
q = 0.05
quantiles = []
for i in range(10):
	quantiles.append(np.quantile(gene_expression.values.T, q, axis = 1))
	q += 0.1

quantiles.append(np.quantile(gene_expression.values.T, 1, axis = 1)+1)
quantiles = np.array(quantiles)
quantiles = np.mean(quantiles, axis = 1)

# Ryan's adaptation of Hanna's random subsampler
def random_gene_subsetting(size):
	cell_cycle_bin = []
	for i in range(size):
		loc = np.where(np.mean(gene_expression.loc[cell_cycle[i]]) > quantiles)[0]
		cell_cycle_bin.append(loc[-1])

	# Take mean expression for finding other genes in bin
	gene_expression_mean = pd.DataFrame(np.mean(gene_expression.values, axis = 1), index = gene_expression.index, columns = ["Mean_exp"])

	# For each cell cycle gene - find the bin, sample a random gene from that same bin
	rand_gene_list = []
	for i in cell_cycle_bin:
		range_min = quantiles[i]
		range_max = quantiles[i + 1]
		random_possibilities = gene_expression_mean[(gene_expression_mean["Mean_exp"] > range_min) & (gene_expression_mean["Mean_exp"] < range_max)]
		random_possibilities_index = random_possibilities.index

		rand_gene_index = np.random.randint(0, random_possibilities_index.shape[0], size=1)
		rand_gene = random_possibilities_index[rand_gene_index][0]
		rand_gene_list.append(rand_gene)

	return rand_gene_list

num_genes = 4
num_groups = 10
group_genes = []
for _ in range(num_groups):
	group_genes.append(random_gene_subsetting(num_genes))

for ind, genes in enumerate(group_genes):
	rows_to_drop = [name for name in gene_expression.index if name not in genes]
	df = gene_expression.drop(labels=rows_to_drop)

	with open("random_genes_"+str(ind)+".txt", "w") as f:
		for gene in genes:
			f.write(gene)
			f.write("\n")

	mat = df.to_numpy().T
	np.save("random_proj_"+str(ind)+".npy", mat)
