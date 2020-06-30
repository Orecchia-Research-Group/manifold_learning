import numpy as np
from scipy.spatial import Delaunay

class AugmentedDelaunay:
	def __init__(self, points, point_names):
		if not isinstance(points, np.ndarray):
			raise TypeError("points should be a NumPy array; currently of type " + str(type(points)))
		if len(points.shape) != 2:
			raise ValueError("points should be a 2D NumPy array")
		self.delaunay = Delaunay(points)
		self.size = points.shape[0]
		assert isinstance(point_names, list)
		assert all(isinstance(x, int) for x in point_names)
		assert len(point_names) == self.size
