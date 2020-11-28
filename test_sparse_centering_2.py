import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances as euclid
import scipy.sparse
from manifold_utils.mSVD import hypersphere, eigen_calc_from_dist_mat, get_centered_sparse, Sparse_eigen_calc_from_dist_mat_uncentered
from manifold_utils.iga import iga

points = np.vstack([hypersphere(2, 100).T, np.zeros((98, 100))]).T
points += 1
points_sparse = scipy.sparse.csr_matrix(points)
dist_mat = euclid(points)

routine = Sparse_eigen_calc_from_dist_mat_uncentered
xbar = np.ones(100)

radii, _, eigvals, eigvecs = routine(points_sparse, xbar, dist_mat, 0, np.max(dist_mat[0, :]))
tru_radii, tru_eigvals, tru_eigvecs = eigen_calc_from_dist_mat(points, dist_mat, 0)

clipped_tru_radii = tru_radii[-len(radii):]
diffs = np.array(radii) - np.array(clipped_tru_radii)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(diffs)

#plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

eigvals = np.stack(eigvals, axis=0)
#print(eigvals.shape)
for j in range(eigvals.shape[1]):
	ax1.plot(eigvals[:, j])

tru_eigvals = np.stack(tru_eigvals, axis=0)
for j in range(tru_eigvals.shape[1]):
	ax2.plot(tru_eigvals[:, j])

#plt.show()

tru_eigvecs = tru_eigvecs[-len(eigvecs):]
eigvecs = np.stack(eigvecs, axis=0)
tru_eigvecs = np.stack(tru_eigvecs, axis=0)

igas = [eigvecs[j, :, :].dot(tru_eigvecs[j, :, -10:]) for j in range(eigvecs.shape[0])]
dists = []
for azalia in igas:
	_, s, _ = np.linalg.svd(azalia)
	#print(180 * np.arccos(s) / np.pi)
	dist = np.sum(np.square(np.arccos(s)))
	dists.append(dist)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(dists)

#plt.show()
