import numpy as np
from tqdm import tqdm
from manifold_utils.mSVD import hypersphere, eigen_plot, eigen_calc_from_dist_mat

s9 = hypersphere(1000, 10).T
dist_mat = np.zeros((1000, 1000))
for j in tqdm(range(1000)):
	for k in range(1000):
		if j != k:
			dist_mat[j, k] = np.linalg.norm(s9[j, :] - s9[k, :])

radii, eigval_list, _ = eigen_calc_from_dist_mat(s9, dist_mat, 0)

rmin = radii[0]
rmax = radii[-1]

eigen_plot(eigval_list, radii, rmin, rmax)
