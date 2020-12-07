from time import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import vstack
from sklearn.metrics.pairwise import euclidean_distances as euclid
from ilc_data.ilc_loader import get_sct_var_sparse, get_index_to_gene, get_cell_topic_weights
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
sparse_var = get_sct_var_sparse()
sparse_var_17 = vstack([sparse_var[ind] for ind in inds_17])

dist_mat_17 = euclid(sparse_var_17)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(dist_mat_17[0, :], bins=100)

plt.show()

radius_list, numPoints_list, eigval_list, eigvec_list = sparse_uncentered_msvd(sparse_var_17, dist_mat_17, 0, np.max(dist_mat_17))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(radius_list, eigval_list)

plt.show()
