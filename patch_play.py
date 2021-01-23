import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm
import PatchExtractTools as pet
from combined_mls_pca import rapid_mls_pca
from manifold_utils.mSVD import rapid_eigen_calc_from_dist_mat as eigen_calc

#load matrix
patches = np.load('Denoised3x3Patches.npy')
n_patches = patches.shape[0]
n_landmarks = 100

# Compute distance matrix
dist_mat = euclidean_distances(patches)

# Randomly select indices for landmarks
indices = np.random.choice(n_patches, size=n_landmarks, replace=False)
np.save("data/natural_images/indices.npy", indices)

for ind in tqdm(indices):
	radii, eigval_list, eigvec_list = eigen_calc(patches, dist_mat, ind, k=5, radint=0.01)
	np.save("data/natural_images/radii_"+str(ind)+".npy", np.array(radii))
	np.save("data/natural_images/eigvals_"+str(ind)+".npy", np.stack(eigval_list, axis=0))
	np.save("data/natural_images/eigvecs_"+str(ind)+".npy", np.stack(eigvec_list, axis=0))
