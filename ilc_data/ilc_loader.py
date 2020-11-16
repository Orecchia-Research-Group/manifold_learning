import os
import numpy as np
import scanpy as sc

# Load data
data_dir = os.path.join('.', 'data')
adata_variable_fname = os.path.join(data_dir, 'sct_variable.h5ad')
adata = sc.read(adata_variable_fname)
