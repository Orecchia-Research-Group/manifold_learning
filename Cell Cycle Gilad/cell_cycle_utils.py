from random import sample
import numpy as np
from ripser import Rips, ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
import bats
from tqdm import tqdm

def get_umap(data, pca_n=4):
	pca = PCA(n_components=pca_n)
	pca.fit(data)
	X = pca.transform(data)

	reducer = umap.UMAP(random_state=42, n_components=2)
	embedding = reducer.fit_transform(X)

	return embedding

def birth_lifetime(H_1, ax, y_max=50):
	# color - "blue", "red", "orange", "purple", "green", "yellow"
	# bd - birth-death pairs for lifetime
	lifetime = H_1[:, 1] - H_1[:, 0]

	ax.plot(H_1[:, 0], lifetime, c="orange", marker=".", linewidth=0)
	ax.set_ylabel('Lifetime')
	ax.set_ylim((-1,y_max))
	ax.set_xlabel('Birth')
