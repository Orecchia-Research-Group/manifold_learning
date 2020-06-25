import numpy as np
from pymanopt.manifolds import Sphere

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
