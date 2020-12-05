from time import time
import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from manifold_utils.mSVD import get_centered_sparse
from ilc_data.ilc_loader import get_sct_var_sparse, get_index_to_gene

sparse_var = get_sct_var_sparse()
lin_op = get_centered_sparse(sparse_var, sparse_var.mean(axis=0))

u, s, vt = svds(lin_op, 2)

# Score gene by relative magnitude in global PCA
mags = np.sum(np.square(vt.T), axis=1)
sorted_mags = np.sort(mags)[::-1]
indices = list(range(len(mags)))
indices.sort(key=lambda x: mags[x], reverse=True)

# Create bar plot of top genes
num_bars = 30
ind_to_gene = get_index_to_gene()
indices_to_plot = indices[:num_bars]
mags_to_plot = sorted_mags[:num_bars]
labels_to_plot = [ind_to_gene[ind] for ind in indices_to_plot]

fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(111)
ax.bar(list(range(num_bars)), mags_to_plot, tick_label=labels_to_plot)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=6)

fig.savefig("figures/global_pca.pdf")
