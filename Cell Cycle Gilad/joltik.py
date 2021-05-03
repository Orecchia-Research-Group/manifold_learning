import numpy as np
from ripser import Rips, ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
from tqdm import tqdm

# Hanna's visualization func
def visualize(data, pca_n, plot_label, hue):
	pca = PCA(n_components=pca_n)
	pca.fit(data.T)
	X = pca.transform(data.T)

	reducer = umap.UMAP(random_state=42, n_components=2)
	embedding = reducer.fit_transform(X)

	reducer_3 = umap.UMAP(random_state=42, n_components=3)
	embedding_3 = reducer_3.fit_transform(X)

	plt.figure(figsize=(8, 6))

	sns.scatterplot(embedding[:,0], embedding[:,1], hue = np.ones(embedding.shape[0]), s = 20, linewidth=0, palette = [color_selection[hue]])

	plt.ylabel('UMAP2')
	plt.xlabel('UMAP1')
	plt.title(plot_label)
	plt.legend(bbox_to_anchor=(1.2, 1),borderaxespad=0)

	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.scatter3D(embedding_3[:,0], embedding_3[:,1], embedding_3[:,2], c=color_selection[hue])

	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.scatter3D(embedding_3[:,1], embedding_3[:,2], embedding_3[:,0], c=color_selection[hue])

	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.scatter3D(embedding_3[:,2], embedding_3[:,0], embedding_3[:,1], c=color_selection[hue])

n_randos = 10

def visualize_umap(data, ax, pca_n=4):
	pca = PCA(n_components=pca_n)
	pca.fit(data)
	X = pca.transform(data)

	reducer = umap.UMAP(random_state=42, n_components=2)
	embedding = reducer.fit_transform(X)

	#reducer_3 = umap.UMAP(random_state=42, n_components=3)
	#embedding_3 = reducer_3.fit_transform(X)

	ax.scatter(embedding[:, 0], embedding[:, 1])

for j in tqdm(range(n_randos)):
	mat = np.load("random_proj_"+str(j)+".npy")

	# Make figure for export
	fig = plt.figure(figsize=(10, 5))

	# Perform 2D UMAP on 4 genes of interest, and plot
	ax1 = fig.add_subplot(121)
	visualize_umap(mat, ax1)

	# Plot persistence diagram
	ax2 = fig.add_subplot(122)
	dgms = ripser(mat, maxdim=1)["dgms"]
	plot_diagrams(dgms, ax=ax2)

	fig.savefig("rando_caricature_"+str(j)+".png")
