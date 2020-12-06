from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

### Load in breast cancer test data
breast = load_breast_cancer()
breast_data = breast.data
breast_data_norm = StandardScaler().fit_transform(breast_data)

### Implement sklearn PCA on breast cancer test data
pca_breast = PCA(n_components=2)
PCs_breast = pca_breast.fit_transform(breast_data_norm)
sk_eigenvecs = pca_breast.components_.T

### Implement sklearn PCA on breast cancer test data (Unnormalized)
pca_breast = PCA(n_components=2)
PCs_breast_wild = pca_breast.fit_transform(breast_data)
sk_eigenvecs_wild = pca_breast.components_.T

### Implement scanpy.tl PCA on breast cancer test data
import scanpy as sc
_, pre_eigenvecs, _, _ = sc.tl.pca(breast_data_norm, n_comps=2, return_info=True)
scanpy_eigenvecs = pre_eigenvecs.T

### Implement scanpy.tl PCA on breast cancer test data (Unnormalized)
import scanpy as sc
_, pre_eigenvecs, _, _ = sc.tl.pca(breast_data, n_comps=2, return_info=True)
scanpy_eigenvecs_wild = pre_eigenvecs.T

### Implement scanpy.pp PCA on breast cancer test data
import scanpy as sc
_, pre_eigenvecs, _, _ = sc.pp.pca(breast_data_norm, n_comps=2, return_info=True)
scanpy_eigenvecs_pp = pre_eigenvecs.T

### Implement scanpy.pp PCA on breast cancer test data (Unnormalized)
import scanpy as sc
_, pre_eigenvecs, _, _ = sc.pp.pca(breast_data, n_comps=2, return_info=True)
scanpy_eigenvecs_pp_wild = pre_eigenvecs.T

### Implement custom PCA on breast cancer test data
from manifold_utils.mSVD import get_centered_sparse
from scipy.sparse.linalg import svds
breast_lin_op = get_centered_sparse(breast_data_norm, np.mean(breast_data_norm, axis=0))

u, s, vt = svds(breast_lin_op, k=2)

our_eigenvecs = vt.T

### Implement custom PCA on breast cancer test data (Unnormalized)
from manifold_utils.mSVD import get_centered_sparse
from scipy.sparse.linalg import svds
breast_lin_op = get_centered_sparse(breast_data, np.mean(breast_data, axis=0))

u, s, vt = svds(breast_lin_op, k=2)

our_eigenvecs_wild = vt.T

### Compare eigenvectors
methods_dict = {}
methods_dict["sk"] = (sk_eigenvecs, sk_eigenvecs_wild)
methods_dict["scanpy_tl"] = (scanpy_eigenvecs, scanpy_eigenvecs_wild)
methods_dict["scanpy_pp"] = (scanpy_eigenvecs_pp, scanpy_eigenvecs_pp_wild)
methods_dict["our"] = (our_eigenvecs, our_eigenvecs_wild)

keys = ["sk", "scanpy_tl", "scanpy_pp", "our"]
num_keys = len(keys)

fig = plt.figure(figsize=(10, 10))
for j, key_a in enumerate(keys):
	eigvecs_a, wild_a = methods_dict[key_a]
	for k, key_b in enumerate(keys):
		eigvecs_b, wild_b = methods_dict[key_b]
		_, s, _ = np.linalg.svd(eigvecs_a.T @ eigvecs_b)
		_, s_wild, _ = np.linalg.svd(wild_a.T @ wild_b)

		ax = fig.add_subplot(num_keys, num_keys, num_keys*j + k + 1)
		ax.plot(s.tolist() + s_wild.tolist(), "ks")
		ax.set_xticks([])
		ax.axhline(-1)
		ax.axhline(1)

plt.show()
