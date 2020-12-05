import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tqdm import tqdm
from manifold_utils.mSVD import eigen_plot
from manifold_utils.iga import iga, arccos_catch_nan

indices = [2, 3, 5, 10, 8, 13, 15, 16]
num_inds = len(indices)
radii_list = []
eigval_list = []
eigvec_list = []
for j in indices:
	radii_list.append(np.load("radii_"+str(j)+".npy"))
	eigval_list.append(np.load("eigvals_"+str(j)+".npy"))
	eigvecs = np.load("eigvecs+"+str(j)+".npy")
	eigvec_list.append(np.swapaxes(eigvecs, 1, 2))

Rmins = [37.5, 36,   37.5, 38,   37,   37, 38, 38]
Rmaxs = [39,   37.5, 39,   39.5, 38.5, 39, 40, 40]

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

cmap = plt.get_cmap("binary")
names = ["Grassmann Distance", "Mean Jordan (Degrees)", "Max Jordan (Degrees)", "Min Jordan (Degrees)"]
#normalize_grass = Normalize(0, np.sqrt(5)*np.pi/2)
normalize_grass = None
#normalize_jordan = Normalize(40, 90)
normalize_jordan = Normalize(0, 90)
normalize_dict = dict(zip(names, [normalize_grass, normalize_jordan, normalize_jordan, normalize_jordan]))

for k in [1, 2, 3, 4, 5]:
	azalias = []
	for eigvecs, Rmin_ind, Rmax_ind in zip(eigvec_list, Rmin_inds, Rmax_inds):
		top_k = eigvecs[Rmin_ind:Rmax_ind, :, -k:]
		hyperplanes = [top_k[j, :, :] for j in range(top_k.shape[0])]
		azalias.append(iga(hyperplanes))

	mat_grass = np.zeros((num_inds, num_inds))
	mat_mean = np.zeros((num_inds, num_inds))
	mat_max = np.zeros((num_inds, num_inds))
	mat_min = np.zeros((num_inds, num_inds))
	for j, az_j in tqdm(enumerate(azalias)):
		for jj, az_jj in enumerate(azalias):
			mat_prod = az_j.T @ az_jj
			_, s, _ = np.linalg.svd(mat_prod)
			jord_rad = arccos_catch_nan(s)
			mat_mean[j, jj] = 180 * np.mean(jord_rad) / np.pi
			mat_max[j, jj] = 180 * np.max(jord_rad) / np.pi
			mat_min[j, jj] = 180 * np.min(jord_rad) / np.pi
			mat_grass[j, jj] = np.sqrt(np.sum(np.square(jord_rad)))

	fig = plt.figure()
	ax1 = fig.add_subplot(221)
	ax2 = fig.add_subplot(222)
	ax3 = fig.add_subplot(223)
	ax4 = fig.add_subplot(224)
	ax1.matshow(mat_grass, cmap=cmap, norm=normalize_grass)
	ax2.matshow(mat_mean, cmap=cmap, norm=normalize_jordan)
	ax3.matshow(mat_max, cmap=cmap, norm=normalize_jordan)
	ax4.matshow(mat_min, cmap=cmap, norm=normalize_jordan)
	ax1.set_ylabel("Grassmann Distance ("+str(k)+")", fontsize=8)
	ax2.set_ylabel("Mean Jordan ("+str(k)+")", fontsize=8)
	ax3.set_ylabel("Max Jordan ("+str(k)+")", fontsize=8)
	ax4.set_ylabel("Min ("+str(k)+")", fontsize=8)
	for ax in [ax1, ax2, ax3, ax4]:
		ax.set_xticklabels(["goober"]+[str(ind) for ind in indices], fontsize=6)
		ax.set_yticklabels(["goober"]+[str(ind) for ind in indices], fontsize=6)
	fig.savefig("k = "+str(k)+".pdf")
