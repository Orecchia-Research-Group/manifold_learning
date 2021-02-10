from time import time
import numpy as np
import ilc_data.ilc_loader as ILC
from tda_utils.density_filtration import sort_distances_per_point, indices_for_density_filtration
from tda_utils.witness_complex import *

# Load in ILC data
sct_sparse = ILC.get_sct_var_scale()
dist_mat = ILC.get_dist_mat()
n, d = sct_sparse.shape
print("n = "+str(n))
print("d = "+str(d))
print("shape of dist_mat: "+str(dist_mat.shape))

# Load in distances per point
distances_per_point = np.load("data/ilc_scale_data_distances_per_point.npy")

# time computation of density filtrations
print("Timing computation of density filtrations...")
density_indices = dict()
density_filtrations = dict()
ks = [10, 100, 1000, 10000]
ps = [0.01, 0.05, 0.1, 0.2]
for k in ks:
	for p in ps:
		print("\tk = "+str(k)+", p = "+str(p)+":")
		start = time()
		indices = indices_for_density_filtration(distances_per_point, k, p)
		density_indices[(k, p)] = indices
		mask = np.zeros(n, dtype=bool)
		mask[indices] = True
		sqr_mask = np.outer(mask, mask)
		density_filtrations[(k, p)] = np.reshape(dist_mat[sqr_mask], (len(indices), len(indices)))
		print("\t"+str(time()-start)+" seconds\n")

print("Timing computation of landmarks...")
landmark_indices = dict()
landmark_mats = dict()
pps = [0.05, 0.1, 0.2]
for k in ks:
	for p in ps:
		for pp in pps:
			print("\tk = "+str(k)+", p = "+str(p)+", pp = "+str(pp)+":")
			start = time()
			landmark_indices[(k, p, pp)], landmark_mats[(k, p, pp)] = choose_landmarks(density_filtrations[(k, p)], pp)
			print("\t"+str(time()-start)+" seconds\n")

print("Printing numbers of landmarks...")
for k in ks:
	for p in ps:
		for pp in pps:
			toople = (k, p, pp)
			print(str(toople)+": "+str(len(landmark_indices[toople])))
			if toople == (10000, 0.2, 0.05):
				landmark_inds = landmark_indices[toople]
				dense_inds = density_indices[(k, p)]
				true_inds = [dense_inds[ind] for ind in landmark_inds]
				to_save = np.vstack([sct_sparse[ind, :] for ind in true_inds])
				np.save("data/sample_landmarks.npy", to_save)
				np.save("data/sample_landmark_indices.npy", true_inds)
