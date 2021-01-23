from time import time
import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from manifold_utils.mSVD import get_centered_sparse
from ilc_data.ilc_loader import get_sct_var_sparse, get_index_to_gene

sparse_var = get_sct_var_sparse()
N, d = sparse_var.shape
lin_op = get_centered_sparse(sparse_var, sparse_var.mean(axis=0))

u, s, vt = svds(lin_op, 50)

lamb = np.square(s)[::-1]/(N - 1)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(lamb)

ax.set_title("Global Scree Plot")
ax.set_xlabel("Eigenvalue Index")
ax.set_ylabel("Eigenvalue")

fig.savefig("global_scree.pdf")
