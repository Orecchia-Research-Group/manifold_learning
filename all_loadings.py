from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from manifold_utils.iga import iga, arccos_catch_nan
from ilc_data.ilc_loader import get_index_to_gene

#indices = [2, 3, 5, 8, 10, 11, 13, 15, 16]
indices = list(range(17))
num_inds = len(indices)

radii_list = []
eigval_list = []
eigvec_list = []
for j in indices:
	radii_list.append(np.load("radii_"+str(j)+".npy"))
	eigval_list.append(np.load("eigvals_"+str(j)+".npy"))
	eigvecs = np.load("eigvecs+"+str(j)+".npy")
	eigvec_list.append(np.swapaxes(eigvecs, 1, 2))

Rminspre = [35.5, 32,   37.5, 36,   35,   37.5, 37.5, 33.5, 38,   38, 37,   36, 42, 37, 40, 38, 38]
Rmaxspre = [37.5, 33.5, 39,   37.5, 37.5, 39,   38,   34,   39.5, 39, 38.5, 39, 44, 39, 42, 40, 40]
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
	# Looking at top two eigvecs
	top_two = eigvecs[Rmin_ind:Rmax_ind, :, -2:]
	hyperplanes = [top_two[j, :, :] for j in range(top_two.shape[0])]
	azalias.append(iga(hyperplanes))

ind_to_gene = get_index_to_gene()
num_genes = 3000

def plot_loadings(ax, azalia, num_loadings=15):
	gene_indices = list(range(num_genes))
	mags = np.sqrt(np.sum(np.square(azalia), axis=1))
	sorted_mags = np.sort(mags)[::-1]
	gene_indices.sort(key=lambda x: mags[x], reverse=True)

	inds_for_loading = gene_indices[:num_loadings]
	genes_for_loading = [ind_to_gene[ind] for ind in inds_for_loading]
	for ind, gene in zip(inds_for_loading, genes_for_loading):
		ax.plot(azalia[ind, 0], azalia[ind, 1], "bo", ms=5)
		ax.annotate(gene, (azalia[ind, 0], azalia[ind, 1]))
	ax.axvline(0, color="k")
	ax.axhline(0, color="k")

for macro_ind, azalia in zip(indices, azalias):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	plot_loadings(ax, azalia)
	ax.set_title("Cheerio "+str(macro_ind))
	fig.savefig("loadings/"+str(macro_ind)+".jpg")
	plt.close(fig)
