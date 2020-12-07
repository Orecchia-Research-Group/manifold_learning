import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
from ilc_data.ilc_loader import get_index_to_gene

ILC = sc.read("data/sct_variable.h5ad")

var_names = ILC.var_names
ind_to_gene = get_index_to_gene()

for j in range(3000):
	assert ind_to_gene[j] == var_names[j]
