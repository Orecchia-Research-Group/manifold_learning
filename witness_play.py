from time import time
import numpy as np
import covid.covid_loader as covid
from tda_utils.density_filtration import sort_distances_per_point, indices_for_density_filtration
from tda_utils.witness_complex import *

# Load in COVID data
racute_data = covid.get_racute_data()
dist_mat = covid.get_racute_dist_mat()
n, d = racute_data.shape
print("n = "+str(n))
print("d = "+str(d))
print("shape of dist_mat: "+str(dist_mat.shape))

try:
	# Load in distances per point
	distances_per_point = np.load("data/covid/racute_distances_per_point.npy")
except FileNotFoundError:
	# time dist mat preprocessing
	print("Timing dist_mat preprocessing...")
	start = time()
	distances_per_point = sort_distances_per_point(dist_mat)
	print(str(time() - start) + " seconds")
	np.save("data/covid/racute_distances_per_point.npy", distances_per_point)

# time computation of density filtrations
print("Timing computation of density filtrations...")
density_indices = dict()
density_filtrations = dict()
ks = [1, 5, 10]
ps = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
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
			# Number of landmarks completely determined by p, pp
			toople = (k, p, pp)
			print("\t(k, p, pp) = "+str(toople))
			start = time()
			landmark_inds = landmark_indices[toople]
			dense_inds = density_indices[(k, p)]
			true_inds = [dense_inds[ind] for ind in landmark_inds]
			to_save = np.vstack([racute_data[ind, :] for ind in true_inds])
			np.save("data/covid/sample_landmarks_kis"+str(toople)+".npy", to_save)
			np.save("data/covid/sample_landmark_indices_kis"+str(toople)+".npy", true_inds)
			print("\t"+str(time() - start)+" seconds\n")
