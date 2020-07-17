import numpy as np

class point_collection:
	"""
	Takes a n x d NumPy array and stores each row as a separate
	vector. Assigns each vector with an index. Allows for lazy
	sorting by distance given a center point
	"""
	def __init__(self, point_array, dist_mat):
		try:
			assert isinstance(point_array, np.ndarray)
			assert isinstance(dist_mat, np.ndarray)
		except AssertionError:
			raise TypeError("Both point_array and dist_mat should be of type np.ndarray")
		try:
			assert len(point_array.shape) == 2
		except AssertionError:
			raise AttributeError("point_array should be a 2D array; currently has %d axes") % len(point_array.shape)
		try:
			assert len(dist_mat.shape) == 2
		except AssertionError:
			raise AttributeError("dist_mat should be a 2D array; currently has %d axes") % len(dist_mat.shape)
		try:
			assert dist_mat.shape == (point_array.shape[0], ) * 2
		except AssertionError:
			raise AttributeError("If point_array has shape (n, d), then dist_mat should have shape (n, n)")

		self.n = point_array.shape[0]
		self.points = list((j, point_array[j, :]) for j in range(self.n))
		self.dist_mat = dist_mat
		self.remaining_points = set(range(self.n))

	def points_sorted_from_center(self, center_name):
		"""
		returns a (lazily) sorted list of the points sorted in increasing radial
		distance from a center point. Center point is given by
		the integer label center_name.
		"""
		if not isinstance(center_name, int):
			raise TypeError("center_name must be an int")
		if center_name < 0 or center_name >= self.n:
			raise ValueError("center_name should be an integer between 0 and n-1, inclusive")

		center_value = self.points[center_name][1]
		points_by_distance = list(enumerate(self.dist_mat[center_name, :]))
		# TODO: Go back and make list lazy
		points_by_distance.sort(key=lambda x: np.linalg.norm(x[1]-center_value))
		return points_by_distance
