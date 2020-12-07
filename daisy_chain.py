import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tqdm import tqdm
from manifold_utils.iga import iga, arccos_catch_nan
from ilc_data.ilc_loader import get_sct_var_sparse, get_index_to_gene

trail_indices = np.load("data/trail_indices.npy")
sparse_var = get_sct_var_sparse()

indices = list(range(17))
num_inds = len(indices)
true_indices = [trail_indices[j] for j in indices]

center_list = []
radii_list = []
eigval_list = []
eigvec_list = []
for j, j_true in zip(indices, true_indices):
	center_list.append(sparse_var[j_true, :].todense()[0, :])
	radii_list.append(np.load("radii_"+str(j)+".npy"))
	eigval_list.append(np.load("eigvals_"+str(j)+".npy"))
	eigvecs = np.load("eigvecs+"+str(j)+".npy")
	eigvec_list.append(np.swapaxes(eigvecs, 1, 2))

preRmins = [35.5, 32,   37.5, 36,   35,   37.5, 37.5, 33.5, 38,   38, 37,   36, 42, 37, 40, 38, 38]
preRmaxs = [37.5, 33.5, 39,   37.5, 37.5, 39,   38,   34,   39.5, 39, 38.5, 39, 44, 39, 42, 40, 40]
Rmins = [preRmins[j] for j in indices]
Rmaxs = [preRmaxs[j] for j in indices]

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
	top_two = eigvecs[Rmin_ind:Rmax_ind, :, -2:]
	hyperplanes = [top_two[j, :, :] for j in range(top_two.shape[0])]
	azalias.append(iga(hyperplanes))

num_genes = 3000
ind_to_gene = get_index_to_gene()

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

figs = []
axx = []
for macro_ind, azalia in zip(indices, azalias):
	fig = plt.figure()
	figs.append(fig)
	ax = fig.add_subplot(111)
	axx.append(ax)
	plot_loadings(ax, azalia)
	ax.set_title("Cheerio "+str(macro_ind))

for j in range(len(azalias) - 1):
	trans_vec = center_list[j+1] - center_list[j]
	dir_vec = trans_vec / np.linalg.norm(trans_vec)
	azalia_pre = azalias[j]
	azalia_post = azalias[j+1]
	pre_vec = azalia_pre.T @ dir_vec.T
	post_vec = azalia_post.T @ dir_vec.T
	axx[j].arrow(0, 0, pre_vec[0, 0], pre_vec[1, 0], color="r", head_width=0.01, head_starts_at_zero=True, length_includes_head=True)
	axx[j+1].arrow(post_vec[0, 0], post_vec[1, 0], -post_vec[0, 0], -post_vec[1, 0], color="b", head_width=0.01, head_starts_at_zero=True, length_includes_head=True)

for j, fig in zip(indices, figs):
	fig.savefig("chain_gang/"+str(j)+".jpg")
