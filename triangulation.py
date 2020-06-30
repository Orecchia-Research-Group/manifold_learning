import numpy as np
from scipy.spatial import Delaunay
from itertools import chain, combinations

def power_set_bar_empty(some_tuple):
	"""
	Given a tuple, returns all nonempty,
	sorted subtuples of that list, using the ordering
	of the original tuple for sorting
	"""
	# Modified from Mark Rushakoff's answer to
	# the following StackOverflow question:
	# https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
	return list(chain.from_iterable(combinations(some_tuple, r) for r in range(1, len(some_tuple)+1)))

def all_faces_from_d_simplices(simplices):
	assert all(map(lambda x: len(x) == len(simplices[0]), simplices))
	face_set = set()
	for simplex in simplices:
		faces = power_set_bar_empty(simplex)
		for face in faces:
			face_set.add(face)
	return face_set

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
		self.point_names = point_names

	def get_simplices(self):
		simplices = []
		original_simplices = self.delaunay.simplices
		for simplex in original_simplices:
			temp = map(lambda x: point_names[x], simplex)
			temp.sort()
			simplices.append(tuple(temp))
		return simplices
