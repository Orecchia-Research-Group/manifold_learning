from copy import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tqdm import tqdm
from manifold_utils.mSVD import eigen_plot
from manifold_utils.iga import iga, arccos_catch_nan
from ilc_data.ilc_loader import get_radii_scale

indices = list(range(17))
num_inds = len(indices)
radii_list = []
eigval_list = []
eigvec_list = []
for j in indices:
	radii_list.append(np.load("scale_intermediates/radii_"+str(j)+".npy"))
	eigval_list.append(np.load("scale_intermediates/eigvals_"+str(j)+".npy"))
	eigvec_list.append(np.load("scale_intermediates/eigvecs+"+str(j)+".npy"))

Rminspre, Rmaxspre = get_radii_scale()
Rmins = [Rminspre[j] for j in indices]
Rmaxs = [Rmaxspre[j] for j in indices]

Rmin_inds = []
Rmax_inds = []
for radii, Rmin, Rmax in zip(radii_list, Rmins, Rmaxs):
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
	top_two = eigvecs[Rmin_ind:Rmax_ind, :, -3:]
	hyperplanes = [top_two[j, :, :] for j in range(top_two.shape[0])]
	azalias.append(iga(hyperplanes))

mat_grass = np.zeros((num_inds, num_inds))
mat_mean = np.zeros((num_inds, num_inds))
mat_max = np.zeros((num_inds, num_inds))
mat_min = np.zeros((num_inds, num_inds))
for j, az_j in tqdm(enumerate(azalias)):
	for k, az_k in enumerate(azalias):
		if j == k:
			mat_mean[j, k] = np.nan
			mat_max[j, k] = np.nan
			mat_min[j, k] = np.nan
			mat_grass[j, k] = np.nan
		else:
			mat_prod = az_j.T @ az_k
			_, s, _ = np.linalg.svd(mat_prod)
			jord_rad = arccos_catch_nan(s)
			mat_mean[j, k] = 180 * np.mean(jord_rad) / np.pi
			mat_max[j, k] = 180 * np.max(jord_rad) / np.pi
			mat_min[j, k] = 180 * np.min(jord_rad) / np.pi
			mat_grass[j, k] = np.sqrt(np.sum(np.square(jord_rad)))

mat_mean = np.ma.masked_invalid(mat_mean)
mat_max = np.ma.masked_invalid(mat_max)
mat_min = np.ma.masked_invalid(mat_min)
mat_grass = np.ma.masked_invalid(mat_grass)

cmap = copy(plt.get_cmap("binary"))
cmap.set_bad(color="red")

names = ["Grassmann Distance", "Mean Jordan (Degrees)", "Max Jordan (Degrees)", "Min Jordan (Degrees)"]
#mat_grass_post = mat_grass.copy()
#for j in range(mat_grass_post.shape[0]):
#	for k in range(mat_grass_post.shape[1]):
#		if mat_grass_post[j, k] < 1e-4:
#			mat_grass_post[j, k] = 0
#mat_mean_post = mat_mean.copy()
#for j in range(mat_mean_post.shape[0]):
#	for k in range(mat_mean_post.shape[1]):
#		if mat_mean_post[j, k] < 1e-4:
#			mat_mean_post[j, k] = 0
#mat_max_post = mat_max.copy()
#for j in range(mat_max_post.shape[0]):
#	for k in range(mat_max_post.shape[1]):
#		if mat_max_post[j, k] < 1e-4:
#			mat_max_post[j, k] = 0
#mat_min_post = mat_min.copy()
#for j in range(mat_min_post.shape[0]):
#	for k in range(mat_min_post.shape[1]):
#		if mat_min_post[j, k] < 1e-4:
#			mat_min_post[j, k] = 0
ultimin_grass = np.unique(mat_grass)[1]
normalize_grass = Normalize(ultimin_grass, np.max(mat_grass))
ultimin_mean = np.unique(mat_mean)[1]
normalize_mean = Normalize(ultimin_mean, np.max(mat_mean))
ultimin_max = np.unique(mat_max)[1]
normalize_max = Normalize(ultimin_max, np.max(mat_max))
ultimin_min = np.unique(mat_min)[1]
normalize_min = Normalize(ultimin_min, np.max(mat_max))
normalize_dict = dict(zip(names, [normalize_grass, normalize_mean, normalize_max, normalize_min]))
for name, mat in zip(names, [mat_grass, mat_mean, mat_max, mat_min]):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	handle = ax.imshow(mat, cmap=cmap, norm=normalize_dict[name])
	ax.set_title(name)
	ax.set_xticks(list(range(17)))
	ax.set_yticks(list(range(17)))
	fig.colorbar(handle)
	fig.savefig("jordan_scale/" + name + ".pdf")
	plt.close(fig)
