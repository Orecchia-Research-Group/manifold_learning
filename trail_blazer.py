from time import time
import numpy as np
from ilc_data.ilc_loader import *
from manifold_utils.mSVD import Sparse_eigen_calc_from_dist_mat_uncentered

trail_indices = np.load("data/trail_indices.npy")

sparse_var = get_sct_var_sparse()
dist_mat = get_dist_mat()

for j, ind in enumerate(trail_indices):
	start = time()
	radii, _, eigval_list, eigvec_list = Sparse_eigen_calc_from_dist_mat_uncentered(sparse_var, dist_mat, ind, np.max(dist_mat), k=10, radint=0.1)
	np.save("radii_"+str(j)+".npy", np.array(radii))
	np.save("eigvals_"+str(j)+".npy", np.stack(eigval_list, axis=0))
	np.save("eigvecs+"+str(j)+".npy", np.stack(eigvec_list, axis=0))
	print("Time elapsed: "+str(time() - start))
