import numpy as np
from manifold_utils.mSVD import eigen_plot
from manifold_utils.iga import iga

num_inds = 17

radii_list = []
eigval_list = []
eigvec_list = []
for j in range(num_inds):
	radii_list.append(np.load("radii_"+str(j)+".npy"))
	eigval_list.append(np.load("eigvals_"+str(j)+".npy"))
	eigvec_list.append(np.load("eigvecs+"+str(j)+".npy"))

for j in range(num_inds):
	eigen_plot(eigval_list[j], radii_list[j], radii_list[j][0], radii_list[j][-1])

Rmin = 36
Rmax = 37.5

Rmin_inds = []
Rmax_inds = []
for radii in radii_list:
	Rmin_dists = np.square(Rmin - radii)
	Rmax_dists = np.square(Rmax - radii)

	Rmin_dist_min = np.min(Rmin_dists)
	Rmin_ind = np.where(Rmin_dists == Rmin_dist_min)
	Rmin_inds.append(Rmin_ind[0][0])

	Rmax_dist_min = np.min(Rmax_dists)
	Rmax_ind = np.where(Rmax_dists == Rmax_dist_min)
	Rmax_inds.append(Rmax_ind[0][0])

azalias = []
for eigvecs, Rmin_ind, Rmax_ind in zip(eigvec_list, Rmin_inds, Rmax_inds):
	# Looking at top two eigvecs
	top_two = eigvecs[:, :, -2:]
	top_two = top_two[Rmin_ind:Rmax_ind, :, :]
	hyperplanes = [top_two[j, :, :] for j in range(top_two.shape[0])]
	azalias.append(iga(hyperplanes))

for j in range(len(azalias) - 1):
	azalia_pre = azalias[j]
	azalia_post = azalias[j+1]
	_, s, _ = np.linalg.svd(azalia_pre.T @ azalia_post)
	print(180 * np.arccos(s) / np.pi)
