import numpy as np
import ilc_data.ilc_loader as ILC
import h5py

# Read in landmark indices
landmark_inds = np.load("data/sample_landmark_indices.npy")
n_landmarks = len(landmark_inds)

# Read in distance matrix
dist_mat = ILC.get_dist_mat()
n_points = dist_mat.shape[0]

# Create "witness slice" of distance matrix
pre_mask = np.array([ind in landmark_inds for ind in range(n_points)])
mask = np.outer(pre_mask, ~pre_mask)
witness_slice = np.reshape(dist_mat[mask], (n_landmarks, n_points-n_landmarks))

print(witness_slice.shape)

#witness_slice.tofile("data/sample_witness_slice.csv", sep=",")

with h5py.File("data/sample_witness_slice.h5", "w") as file:
	file.create_dataset("witness_slice", data=witness_slice)

#arr = np.eye(3)
#with h5py.File("temp.h5", "w") as file:
#	file.create_dataset("temp_name", data=arr)

with h5py.File("data/sample_witness_slice.h5", "r") as file:
	print(dir(file["witness_slice"]))
	print(file["witness_slice"].shape)
