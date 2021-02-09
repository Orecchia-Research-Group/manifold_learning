from time import time
import numpy as np
#from ilc_data.ilc_loader import get_sct_sparse, get_dist_mat
import ilc_data.ilc_loader as ILC
from tda_utils.density_filtration import sort_distances_per_point, indices_for_density_filtration
from tda_utils.witness_complex import *

# Load in ILC data
#sct_sparse = ILC.get_sct_sparse()
# dist_mat = ILC.get_dist_mat()
sct_sparse = ILC.get_sct_var_scale()
dist_mat = ILC.get_dist_mat()
n, d = sct_sparse.shape
print("n = "+str(n))
print("d = "+str(d))
print("shape of dist_mat: "+str(dist_mat.shape))

# time dist mat preprocessing
print("Timing dist_mat preprocessing...")
start = time()
distances_per_point = sort_distances_per_point(dist_mat)
print(str(time() - start) + " seconds")

# time computation of density filtrations
print("Timing computation of density filtrations...")
density_filtrations = dict()
ks = [10, 100, 1000, 10000]
ps = [0.01, 0.05, 0.1, 0.2]
for k in ks:
	for p in ps:
		print("\tk = "+str(k)+", p = "+str(p)+":")
		start = time()
		indices = indices_for_density_filtration(distances_per_point, k, p)
		mask = np.zeros(n, dtype=bool)
		mask[indices] = True
		sqr_mask = np.outer(mask, mask)
		density_filtrations[(k, p)] = np.reshape(dist_mat[sqr_mask], (len(indices), len(indices)))
		print("\t"+str(time()-start)+" seconds\n")

print("Timing computation of landmarks...")
landmark_mats = dict()
pps = [0.05]
for k in ks:
	for p in ps:
		for pp in pps:
			print("\tk = "+str(k)+", p = "+str(p)+", pp = "+str(pp)+":")
			start = time()
			_, landmark_mats[(k, p, pp)] = choose_landmarks(density_filtrations[(k, p)], pp)
			print("\t"+str(time()-start)+" seconds\n")

print("Timing computation of simplex trees...")
simplex_trees = dict()
for k in ks:
	for p in ps:
		for pp in pps:
			print("\tk = "+str(k)+", p = "+str(p)+", pp = "+str(pp)+":")
			start = time()
			simplex_trees[(k, p, pp)] = complex_from_witness_dist_mat(landmark_mats[(k, p, pp)])
			print("\t"+str(time()-start)+" seconds\n")
