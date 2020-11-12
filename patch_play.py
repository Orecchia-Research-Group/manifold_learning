import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from time import time
import PatchExtractTools as pet
from combined_mls_pca import mls_pca

#load matrix
Patches = np.load('Denoised3x3Patches.npy')
dist_mat = euclidean_distances(Patches)
#do ev plot around c given by cid
cid = 5000
#Image_ev_plot(Patches, dist_mat, cid)

start_time = time()
mls_pca(Patches, cid, 2, dist=dist_mat)
runtime = time() - start_time

with open("timestamp.txt", "w") as f:
	f.write(str(runtime))
