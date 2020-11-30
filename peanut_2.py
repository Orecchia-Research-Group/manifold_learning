import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.cm import Greys_r
from matplotlib.colors import Normalize
from tqdm import tqdm
from manifold_utils.mSVD import eigen_plot
from manifold_utils.iga import iga, arccos_catch_nan

# num_inds = 6
num_inds = 17
to_excise = [0, 1, 4, 6, 7, 9]
radii_list = []
eigval_list = []
eigvec_list = []
for j in range(num_inds):
	if j not in to_excise:
		radii_list.append(np.load("radii_"+str(j)+".npy"))
		eigval_list.append(np.load("eigvals_"+str(j)+".npy"))
		eigvecs = np.load("eigvecs+"+str(j)+".npy")
		eigvec_list.append(np.swapaxes(eigvecs, 1, 2))

#Rmins = [35.5, 32,   37.5, 36,   35,   37.5, 37.5, 33.5, 38,   38, 37,   36, 42, 37, 40, 38, 38]
#Rmaxs = [37.5, 33.5, 39,   37.5, 37.5, 39,   38,   34,   39.5, 39, 38.5, 39, 44, 39, 42, 40, 40]
Rmins = [37.5, 36,   37.5, 38,   37,   36, 42, 37, 40, 38, 38]
Rmaxs = [39,   37.5, 39,   39.5, 38.5, 39, 44, 39, 42, 40, 40]

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
	# Looking at top two eigvecs
#	top_two = eigvecs[:, :, -2:]
#	top_two = eigvecs[Rmin_ind:Rmax_ind, :, :]
	top_two = eigvecs[Rmin_ind:Rmax_ind, :, -5:]
	hyperplanes = [top_two[j, :, :] for j in range(top_two.shape[0])]
	azalias.append(iga(hyperplanes))

#for j in range(len(azalias) - 1):
#	azalia_pre = azalias[j]
#	azalia_post = azalias[j+1]
#	_, s, _ = np.linalg.svd(azalia_pre.T @ azalia_post)
#	print(180 * np.arccos(s) / np.pi)

mat_grass = np.zeros((num_inds, num_inds))
mat_mean = np.zeros((num_inds, num_inds))
mat_max = np.zeros((num_inds, num_inds))
for j, az_j in tqdm(enumerate(azalias)):
	for k, az_k in enumerate(azalias):
		mat_prod = az_j.T @ az_k
		_, s, _ = np.linalg.svd(mat_prod)
		jord_rad = arccos_catch_nan(s)
		mat_mean[j, k] = 180 * np.mean(jord_rad) / np.pi
		mat_max[j, k] = 180 * np.max(jord_rad) / np.pi
		mat_grass[j, k] = np.sqrt(np.sum(np.square(jord_rad)))

cmap = plt.get_cmap("binary")

names = ["Grassmann Distance", "Mean Jordan (Degrees)", "Max Jordan (Degrees)"]
normalize_grass = Normalize(0, np.sqrt(5)*np.pi/2)
#normalize_jordan = Normalize(0, 90)
normalize_jordan = Normalize(45, 90)
normalize_dict = dict(zip(names, [normalize_grass, normalize_jordan, normalize_jordan]))
for name, mat in zip(names, [mat_grass, mat_mean, mat_max]):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	handle = ax.imshow(mat, cmap=cmap, norm=normalize_dict[name])
	ax.set_title(name)
	fig.colorbar(handle)
	fig.savefig(name + ".pdf")
