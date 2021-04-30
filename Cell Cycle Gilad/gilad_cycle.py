import numpy as np
import bats
from sklearn.metrics.pairwise import euclidean_distances as euclid
from time import time
from random import sample, seed

pre_data = np.load("gene_prime.npy")
try:
	dist_mat = np.load("gene_prime_dist_mat.npy")
except FileNotFoundError:
	print("Populating dist_mat...")
	dist_mat = euclid(pre_data)
	np.save("gene_prime_dist_mat.npy", dist_mat)
	print("Done")

cutoff = 5.0

start = time()

cp_data = pre_data.copy()
cp_mat = dist_mat.copy()
while True:
	for j in range(cp_mat.shape[0]):
		cp_mat[j, j] = np.inf

	min_dists = []
	for j in range(cp_mat.shape[0]):
		min_dists.append(np.min(cp_mat[j, :]))

	min_ind = np.argmin(min_dists)
	min_val = min_dists[min_ind]

	if min_val <= cutoff:
		cp_data = np.vstack([cp_data[:min_ind, :], cp_data[(min_ind + 1):, :]])
		cp_mat = euclid(cp_data)
	else:
		break

print(time() - start)
print(cp_data.shape)

# Fix RNG
seed(42)
keepsize = 334
indices = list(range(cp_data.shape[0]))
kept_indices = sample(indices, keepsize)

pre_data = cp_data[kept_indices, :]

if True:
	np.save("trimmed_data.npy", pre_data)

data = bats.DataSet(bats.Matrix(pre_data))
dist = bats.Euclidean()

rf = bats.RipsFiltration(data, dist, np.inf, 2)

start = time()
rc = bats.reduce(rf, bats.F2())
print(time() - start)
