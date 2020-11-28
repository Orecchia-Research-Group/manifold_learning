import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances as euclid
import scipy.sparse
from manifold_utils.mSVD import hypersphere, eigen_calc_from_dist_mat, get_centered_sparse, Sparse_eigen_calc_from_dist_mat_uncentered

points = hypersphere(100, 100).T
points += 1
points_sparse = scipy.sparse.csr_matrix(points)
dist_mat = euclid(points)

routine = Sparse_eigen_calc_from_dist_mat_uncentered
xbar = np.ones(100)

radii, _, eigvals, eigvecs = routine(points_sparse, xbar, dist_mat, 0, np.max(dist_mat[0, :]))
tru_radii, tru_eigvals, tru_eigvecs = eigen_calc_from_dist_mat(points, dist_mat, 0)
