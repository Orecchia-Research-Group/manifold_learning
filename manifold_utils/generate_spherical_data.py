import os
from tqdm import tqdm
import numpy as np
from pymanopt.manifolds import Sphere

# Remove pre-existing files
try:
	os.remove("data/spherical_toy_data.npy")
except FileNotFoundError:
	pass
try:
	os.remove("data/spherical_toy_dist.npy")
except FileNotFoundError:
	pass

# Generate 1,000 points from S^9
n = 1000
mfld = Sphere(10)

points = []
for _ in range(n):
	points.append(mfld.rand())
points = np.stack(points, axis=0)

# Add Gaussian noise
points += 0.1*np.random.randn(1000, 10)

np.save("data/spherical_toy_data", points)

# Populate distance matrix
dist_mat = np.zeros((n, n))
print("Populating distance matrix...")
for j in tqdm(range(n)):
	for k in range(n):
		dist_mat[j, k] = np.linalg.norm(points[j, :] - points[k,:])
np.save("data/spherical_toy_dist", dist_mat)
