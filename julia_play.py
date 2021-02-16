import numpy as np
import ilc_data.ilc_loader as ILC
import h5py

ks = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000]

for k in ks:
	strk = str(k)
	# Read in landmark indices
	landmark_inds = np.load("data/sample_landmark_indices_kis"+strk+".npy")
	n_landmarks = len(landmark_inds)

	# Read in distance matrix
	dist_mat = ILC.get_dist_mat()
	n_points = dist_mat.shape[0]

	# Create "witness slice" of distance matrix
	pre_mask = np.array([ind in landmark_inds for ind in range(n_points)])
	mask = np.outer(pre_mask, ~pre_mask)
	witness_slice = np.reshape(dist_mat[mask], (n_landmarks, n_points-n_landmarks))

	np.save("data/sample_witness_slice_kis"+strk+".npy", witness_slice)
