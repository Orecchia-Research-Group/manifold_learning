import numpy as np
import ilc_data.ilc_loader as ILC
import h5py
from tqdm import tqdm

#ks = [1, 5, 10, 50]
ks = [5]
ps = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
pps = [0.05, 0.1, 0.2]

for k in tqdm(ks):
	for p in ps:
		for pp in pps:
			str_toople = str((k, p, pp))
			# Read in landmark indices
			landmark_inds = np.load("data/sample_landmark_indices_kis"+str_toople+".npy")
			n_landmarks = len(landmark_inds)

			# Read in distance matrix
			dist_mat = ILC.get_dist_mat()
			n_points = dist_mat.shape[0]

			# Create "witness slice" of distance matrix
			pre_mask = np.array([ind in landmark_inds for ind in range(n_points)])
			mask = np.outer(pre_mask, ~pre_mask)
			witness_slice = np.reshape(dist_mat[mask], (n_landmarks, n_points-n_landmarks))

			np.save("data/sample_witness_slice_kis"+str_toople+".npy", witness_slice)
