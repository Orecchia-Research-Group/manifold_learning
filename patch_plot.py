import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from manifold_utils.mSVD import eigen_plot
from manifold_utils.iga import arccos_catch_nan

# Load landmark indices
indices = np.load("data/natural_images/indices.npy")
indices = indices.astype(int)

def vanilla_eigenplots():
	# vanilla eigenplots
	for ind in tqdm(indices):
		radii = np.load("data/natural_images/radii_"+str(ind)+".npy")
		eigvals = np.load("data/natural_images/eigvals_"+str(ind)+".npy")

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(radii, eigvals)

		fig.savefig("figures/natural_images/eigen_plot_"+str(ind)+".png")
		plt.close(fig)

def grassmann_distance(X, Y):
	"""
	Finds the Grassmann distance between two matrices whose rowspan
	is the represented subspace, as is done in Neretin.

	Remember: As opposed to computing "X.T @ Y", as is done in the
	Neretin paper, this assumes these are transpose, and computes
	"X @ Y.T"
	"""
	_, s, _ = np.linalg.svd(X @ Y.T)
	jord_rad = arccos_catch_nan(s)
	return np.sqrt(np.sum(np.square(jord_rad)))

def jordan_autocorrelation():
	# Jordan "autocorrelation"
	ind = indices[0]
	radii = np.load("data/natural_images/radii_"+str(ind)+".npy")
	eigvals = np.load("data/natural_images/eigvals_"+str(ind)+".npy")
	eigvecs = np.load("data/natural_images/eigvecs_"+str(ind)+".npy")
	eigvecs = eigvecs[:, :2, :]

	delays = [1, 5, 25, 125]
	fig = plt.figure()
	for j, delay in enumerate(delays):
		ax = fig.add_subplot(2, 2, j+1)

		grass_dist = []
		for k in range(eigvals.shape[0]):
			try:
				grass_dist.append(grassmann_distance(eigvecs[k, :, :], eigvecs[k + delay, :, :]))
			except IndexError:
				break
		ax.plot(radii[:len(grass_dist)], grass_dist)
		ax.set_title("delay = "+str(delay), fontsize=6)
	fig.savefig("figures/natural_images/autocorr_"+str(ind)+".png")
	plt.close(fig)

def jordan_matrix():
	for ind in tqdm(indices):
		radii = np.load("data/natural_images/radii_"+str(ind)+".npy")
		eigvals = np.load("data/natural_images/eigvals_"+str(ind)+".npy")
		eigvecs = np.load("data/natural_images/eigvecs_"+str(ind)+".npy")
		eigvecs = eigvecs[:, :2, :]

		n_hyperplanes = len(radii)
		grass_dists = np.zeros((n_hyperplanes, n_hyperplanes))
		for j in range(n_hyperplanes):
			hyperplane_A = eigvecs[j, :, :]
			for k in range(j, n_hyperplanes):
				hyperplane_B = eigvecs[k, :, :]
				grass_dist = grassmann_distance(hyperplane_A, hyperplane_B)
				grass_dists[j, k] = grass_dist
				grass_dists[k, j] = grass_dist

		fig = plt.figure()
		ax = fig.add_subplot(111)
		cmap = plt.get_cmap("binary")

		handle = ax.imshow(grass_dists, cmap=cmap)
		ax.set_title("Point "+str(ind))
		ticklabels = []
		for xtick in ax.get_xticks():
			try:
				ticklabels.append(str(radii[int(xtick)])[:4])
			except IndexError:
				break
		ax.set_xticklabels(ticklabels, rotation=45)
		ax.set_yticklabels(ticklabels, rotation=45)
		fig.colorbar(handle)

		fig.savefig("figures/natural_images/autocorr_matrix_"+str(ind)+".png")
		plt.close(fig)

		fig_bleach = plt.figure()
		ax = fig_bleach.add_subplot(111)
		ax.imshow(grass_dists, cmap=cmap)
		ax.set_xticks([])
		ax.set_yticks([])

		fig_bleach.savefig("figures/natural_images/autocorr_bleach_"+str(ind)+".png")

jordan_matrix()
