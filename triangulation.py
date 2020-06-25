import numpy as np
from scipy.spatial import Delaunay

points = np.load("data/delaunay_test.npy")
temp = Delaunay(points)

print(dir(temp))
print(temp.simplices)
print(temp.ndim)
