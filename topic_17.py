from time import time
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.sparse import vstack
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import euclidean_distances as euclid
from ilc_data.ilc_loader import get_sct_var_scale, get_index_to_gene, get_cell_topic_weights
from manifold_utils.mSVD import Sparse_eigen_calc_from_dist_mat_uncentered as sparse_uncentered_msvd
from manifold_utils.iga import iga

initial_plot = False

topic_weights = get_cell_topic_weights()

arr = topic_weights[:, -1].tolist()
arr.sort(reverse=True)

if initial_plot:
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(arr)

	plt.show()

thresh_17 = 0.05
inds_17 = [ind for ind in range(len(arr)) if arr[ind] >= thresh_17]

# Load in entire point cloud
scale_var = get_sct_var_scale()
scale_var_17 = np.vstack([scale_var[ind] for ind in inds_17])

dist_mat_17 = euclid(scale_var_17)

dist_plot = False
if dist_plot:
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.hist(dist_mat_17[0, :], bins=100)

	plt.show()

cov_x = np.cov(scale_var_17, rowvar=False)
#u, s, vt = svds(scale_var_17, k=3)
s, v = np.linalg.eigh(cov_x)

#v = vt.T

num_genes = 30
ind_to_gene = get_index_to_gene()
for j, r in enumerate([2, 1]):
	indices = list(range(3000))
	mags = np.abs(v[:, r])
	indices.sort(key=lambda x: mags[x], reverse=True)
	inds_to_plot = indices[:num_genes]
	heights_to_plot = [v[ind, r] for ind in inds_to_plot]
	genes_to_plot = [ind_to_gene[ind] for ind in inds_to_plot]

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.bar(list(range(num_genes)), heights_to_plot, tick_label=genes_to_plot)
	ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=6)
	ax.set_title(str(j+1)+"th Eigenvector")
	plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
u = scale_var_17 @ v
ax.scatter(u[:, 2], u[:, 1], u[:, 0])
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_zlabel("PC 3")

plt.show()
"""
radius_list, numPoints_list, eigval_list, eigvec_list = sparse_uncentered_msvd(sparse_var_17, dist_mat_17, 0, np.max(dist_mat_17))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(radius_list, eigval_list)

plt.show()
"""
