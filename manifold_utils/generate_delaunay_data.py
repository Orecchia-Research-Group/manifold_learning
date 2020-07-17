import os
import itertools as it
import numpy as np

# Delete file if pre-existing
try:
	os.remove("data/delaunay_toy.npy")
except FileNotFoundError:
	pass
try:
	os.remove("data/delaunay_test.npy")
except FileNotFoundError:
	pass

# for delaunay_toy: 100 points from Gaussian in R^8
n = 100
d = 8
points = np.random.randn(100, 8)

np.save("data/delaunay_toy", points)

# for delaunay_test: vertices of unit cube in R^3
bins = [-1, 1]
rows = []
for elem in it.product(bins, bins, bins):
	rows.append(np.array(elem))
points = np.stack(rows, axis=0)

np.save("data/delaunay_test", points)
