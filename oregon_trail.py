import numpy as np
from manifold_utils.mSVD import eigen_plot

num = 17
for j in range(num):
	radii = np.load("radii_"+str(j)+".npy")
	eigvals = np.load("eigvals_"+str(j)+".npy")

	eigval_list = [eigvals[j, :] for j in range(eigvals.shape[0])]
	eigen_plot(eigval_list, radii, radii[0], radii[-1])
