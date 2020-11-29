import numpy as np
from ilc_data.ilc_loader import *
from manifold_utils.mSVD import Sparse_eigen_calc_from_dist_mat_uncentered

trail_indices = np.load("data/trail_indices.npy")

seed_ind = trail_indices[0]

points = get_sct_var_sparse()
dist_mat = get_dist_mat()
dist_vec = dist_mat[seed_ind, :]
sorted_vec = np.sort(dist_vec)
indices = list(range(len(sorted_vec)))
indices.sort(key=lambda x: dist_vec[x])

cut_indices = np.array(indices[:50])
np.save("data/closest_50.npy", cut_indices)

for j, ind in enumerate(cut_indices):
	radii, numPoints_list, eigval_list, eigvec_list = Sparse_eigen_calc_from_dist_mat_uncentered(points, dist_mat, ind, 37.5, k=2)
	np.save("data/continuity_radii_"+str(j)+".npy", np.array(radii))
	np.save("data/continuity_numPoints_"+str(j)+".npy", np.array(numPoints_list))
	np.save("data/continuity_eigval_"+str(j)+".npy", np.stack(eigval_list, axis=0))
	np.save("data/continuity_eigvec_"+str(j)+".npy", np.stack(eigvec_list, axis=0))
