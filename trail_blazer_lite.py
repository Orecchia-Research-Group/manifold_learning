from time import time
import numpy as np
from ilc_data.ilc_loader import *
from manifold_utils.mSVD import eigen_calc_from_dist_mat

trail_indices = np.load("data/trail_indices.npy")

sparse_var = get_sct_var_sparse()
points = sparse_var.todense()
dist_mat = get_dist_mat()

start = time()
radii, eigval_list, eigvec_list = eigen_calc_from_dist_mat(points, dist_mat, 0, radint=0.1)
np.save("radii_test.npy", np.array(radii))
np.save("eigvals_test.npy", np.stack(eigval_list, axis=0))
np.save("eigvecs_test.npy", np.stack(eigvec_list, axis=0))
print("Time elapsed: "+str(time() - start))
