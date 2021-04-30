import numpy as np
import umap
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from ripser import Rips, ripser
from persim import plot_diagrams

data = np.load("trimmed_data.npy")

pca = PCA(n_components=4)
pca.fit(data)
X = pca.transform(data)

reducer = umap.UMAP(random_state=42)
embedding_sub = reducer.fit_transform(X)

f = plt.figure()
ax = f.add_subplot(111)
ax.scatter(embedding_sub[:, 0], embedding_sub[:, 1])
ax.set_ylabel('UMAP2')
ax.set_xlabel('UMAP1')
ax.set_title('UMAP - Check')

plt.show()
plt.close(f)

f = plt.figure(figsize=(10, 5))
dgms = ripser(data, maxdim=1)["dgms"]
plot_diagrams(dgms)

plt.show()
