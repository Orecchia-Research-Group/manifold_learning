import numpy as np
import matplotlib.pyplot as plt
from manifold_utils.iga import iga
from ilc_data.ilc_loader import *

num_inds = 17

radii_list = []
eigval_list = []
eigvec_list = []
for j in range(num_inds):
	radii_list.append(np.load("radii_"+str(j)+".npy"))
	eigval_list.append(np.load("eigvals_"+str(j)+".npy"))
	eigvecs = np.load("eigvecs+"+str(j)+".npy")
	eigvec_list.append(np.swapaxes(eigvecs, 1, 2))

Rmins = [35.5, 32,   37.5, 36,   35,   37.5, 37.5, 33.5, 38,   38, 37,   36, 42, 37, 40, 38, 38]
Rmaxs = [37.5, 33.5, 39,   37.5, 37.5, 39,   38,   34,   39.5, 39, 38.5, 39, 44, 39, 42, 40, 40]

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

def tangent_gene_plots(ax1, ax2, azalia, num_amb_vecs=20):
	mags = np.sum(np.square(azalia), axis=1)

	indices = list(range(azalia.shape[0]))
	sorted_mags = np.sort(mags)[::-1]
	indices.sort(key=lambda x: mags[x], reverse=True)

	ax1.bar(list(range(num_amb_vecs)), sorted_mags[:num_amb_vecs],
			tick_label=[ind_to_gene[ind] for ind in indices[:num_amb_vecs]])
#	ax1.set_xticklabels(ax1.get_xticklabels(), rotation="vertical", fontsize=6)
	ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, fontsize=6)
#	ax1.set_ylabel("Magnitude of Projection")
	ax1.set_ylabel("Cheerio "+str(j))

	ax2.plot(sorted_mags)

fig = plt.figure(figsize=(14, 2.5*num_inds))
for j, azalia in enumerate(azalias):
	ax1 = fig.add_subplot(num_inds, 2, 2*j+1)
	ax2 = fig.add_subplot(num_inds, 2, 2*j+2)
	tangent_gene_plots(ax1, ax2, azalia)

fig.savefig("tangent_genes.pdf")
