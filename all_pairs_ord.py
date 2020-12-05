import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from manifold_utils.iga import iga, arccos_catch_nan
from ilc_data.ilc_loader import get_index_to_gene

indices = [2, 3, 5, 8, 10, 11, 13, 15, 16]
num_inds = len(indices)

radii_list = []
eigval_list = []
eigvec_list = []
for j in indices:
	radii_list.append(np.load("radii_"+str(j)+".npy"))
	eigval_list.append(np.load("eigvals_"+str(j)+".npy"))
	eigvecs = np.load("eigvecs+"+str(j)+".npy")
	eigvec_list.append(np.swapaxes(eigvecs, 1, 2))

Rmins = [37.5, 36,   37.5, 37,   38,   36, 37, 38, 38]
Rmaxs = [39,   37.5, 39,   38.5, 39.5, 39, 39, 40, 40]

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
	top_two = eigvecs[Rmin_ind:Rmax_ind, :, -2:]
	hyperplanes = [top_two[j, :, :] for j in range(top_two.shape[0])]
	azalias.append(iga(hyperplanes))

ind_to_gene = get_index_to_gene()

def paired_proj_plots(ax1, ax2, azalia1, azalia2, num_amb_vecs=20, ylabel=""):
	mags1 = np.sqrt(np.sum(np.square(azalia1), axis=1))
	mags2 = np.sqrt(np.sum(np.square(azalia2), axis=1))
	ratio = np.array([(mags1[j]/mags2[j] if mags2[j] != 0 else 0) for j in range(mags1.shape[0])])
	mags = np.array([0 if ratio[j] == 0 else np.max([ratio[j], ratio[j]**(-1)]) for j in range(ratio.shape[0])])
#	to_max = np.stack([ratio, ratio**(-1)], axis=-1)
#	mags = np.max(to_max, axis=1)

	indices = list(range(azalia1.shape[0]))
	sorted_mags = np.sort(mags)[::-1]
	indices.sort(key=lambda x: mags[x], reverse=True)

	ax1.bar(list(range(num_amb_vecs)), sorted_mags[:num_amb_vecs],
		tick_label=[ind_to_gene[ind] for ind in indices[:num_amb_vecs]])
	ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, fontsize=6)
	ax1.set_ylabel(ylabel)

	ax2.plot(sorted_mags)

for j, ind_j in enumerate(indices):
	for k, ind_k in enumerate(indices):
		if j < k:
			fig = plt.figure(figsize=(10, 7))
			ax1 = fig.add_subplot(121)
			ax2 = fig.add_subplot(122)
			ylabel = str(ind_j)+" to "+str(ind_k)
			paired_proj_plots(ax1, ax2, azalias[j], azalias[k], ylabel=ylabel)

			fig.savefig("ords_mag/"+ylabel+".jpg")
			plt.close(fig)
