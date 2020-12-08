from time import time
import numpy as np
from ilc_data.ilc_loader import *
from manifold_utils.mSVD import eigen_calc_from_dist_vec
import matplotlib.pyplot as plt

trail_indices = np.load("data/trail_indices.npy")

scale_var = get_sct_var_scale()
dist_vecs = {}
for ind in trail_indices:
	dist_vecs[ind] = get_dist_vec_scale(ind)

for j, ind in enumerate(trail_indices):
	dist_vec = dist_vecs[ind]
	sorted_vec = np.sort(dist_vec)
	radii, _, eigval_list, eigvec_list = eigen_calc_from_dist_vec(scale_var, dist_vec, sorted_vec[int(np.floor(len(sorted_vec)/10))], k=10, radint=0.01, n_iter=2)
	np.save("scale_intermediates_chibi/radii_"+str(j)+".npy", np.array(radii))
	np.save("scale_intermediates_chibi/eigvals_"+str(j)+".npy", np.stack(eigval_list, axis=0))
	np.save("scale_intermediates_chibi/eigvecs+"+str(j)+".npy", np.stack(eigvec_list, axis=0))
