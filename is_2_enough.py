import numpy as np
import matplotlib.pyplot as plt
from ilc_data.ilc_loader import get_sct_var_scale, get_dist_vec_scale

trail_inds = np.load("data/trail_indices.npy")
base_ind = trail_inds[0]
dist_vec = get_dist_vec_scale(base_ind)

scale_var = get_sct_var_scale()
N, d = scale_var.shape

radii = np.load("scale_intermediates/radii_0.npy")
eigvecs_2 = np.load("scale_intermediates/eigvecs+0.npy")

rad_inds = [30, 35, 40, 45, 50, 55, 60]

jordan_coefs = []
for rad_ind in rad_inds:
	rad = radii[rad_ind]
	points = np.stack([scale_var[ind, :] for ind in range(N) if dist_vec[ind] <= rad], axis=0)
	cov_x = np.cov(points, rowvar=False)
	_, v = np.linalg.eigh(cov_x)
	_, s, _ = np.linalg.svd(v.T @ eigvecs_2[rad_ind])
	jordan_coefs.append(s)

fig = plt.figure()
ax = fig.add_subplot(111)
for j in range(5):
	coefs = [jordan_coef[j] for jordan_coef in jordan_coefs]
	ax.scatter([radii[rad_ind] for rad_ind in rad_inds], coefs)
	ax.axhline(-1)
	ax.axhline(1)

plt.show()
