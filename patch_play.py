import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from time import time
import PatchExtractTools as pet
from combined_mls_pca import rapid_mls_pca
from manifold_utils.mSVD import Sparse_eigen_calc_from_dist_mat_uncentered as eigen_calc

#load matrix
patches = np.load('Denoised3x3Patches.npy')
dist_mat = euclidean_distances(patches)
#do ev plot around c given by cid
cid = 5000

start = time()
radii, _, eigval_list, eigvec_list = eigen_calc(patches, dist_mat, cid, np.max(dist_mat), k=5, radint=0.1)
np.save("data/natural_images/radii_"+str(cid)+".npy", np.array(radii))
np.save("data/natural_images/eigvals_"+str(cid)+".npy", np.stack(eigval_list, axis=0))
np.save("data/natural_images/eigvecs_"+str(cid)+".npy", np.stack(eigvec_list, axis=0))

print("Time elapsed: "+str(time() - start))
