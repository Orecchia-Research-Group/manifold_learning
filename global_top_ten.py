from time import time
import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from manifold_utils.mSVD import get_centered_sparse
from ilc_data.ilc_loader import get_sct_var_sparse, get_index_to_gene

sparse_var = get_sct_var_sparse()
lin_op = get_centered_sparse(sparse_var, sparse_var.mean(axis=0))

u, s, vt = svds(lin_op, 10)
print(s)

print(lin_op @ vt.T)

ind_to_gene = get_index_to_gene()
num_bars = 30
for j in range(10):
	fig = plt.figure(figsize=(15, 5))
	ax = fig.add_subplot(111)
	ax.set_title(str(j+1) + "th Eigenvector")

	eigenvec = vt.T[:, -j]
	mags = np.abs(eigenvec)
	sorted_mags = np.sort(mags)[::-1]
	indices = list(range(len(mags)))
	indices.sort(key=lambda x: mags[x], reverse=True)
	indices_to_plot = indices[:num_bars]
	vals_to_plot = [eigenvec[ind] for ind in indices_to_plot]
	labels_to_plot = [ind_to_gene[ind] for ind in indices_to_plot]

	ax.bar(list(range(num_bars)), vals_to_plot, tick_label=labels_to_plot)
	ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=6)

	fig.savefig("top_eigvecs/"+str(j+1)+"th_eigvec.pdf")
	plt.close(fig)
