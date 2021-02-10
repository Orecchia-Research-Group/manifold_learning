import numpy as np
from sklearn.metrics.pairwise import euclidean_distances as euclid

def get_dist_mat(points):
	"""
	A wrapper for sklearn.metrics.pairwise.euclidean_distance

	inputs:
	points: an n x d NumPy array, where n is the number of samples
		d is the number of points

	returns:
	dist_mat: an n x n matrix whose (i,j)th entry is the distance
	between points i and j
	"""
	return euclid(points)

def sort_distances_per_point(dist_mat):
	"""
	Takes in a n x n matrix of pointwise distances and returns an
	n x (n-1) matrix, the i'th row of which is the distances from
	point i in ascending order (excising the distance from a point
	to itself)
	"""
	n = dist_mat.shape[0]
	rows_to_return = []
	for j in range(n):
		rows_to_return.append(np.sort(dist_mat[j, :]))
	distances_per_point = np.vstack(rows_to_return)
	return distances_per_point[1:, :]

def indices_for_density_filtration(distances_per_point, k, p):
	"""
	Forms a X(k, p) density filtration of the kind used in the
	Carlsson paper.

	inputs:
	distances_per_point: an n x (n-1) matrix, whose i'th row is the
		distances from the i'th point to each other point in
		ascending order.
	k:	a positive integer; a distance parameter which determines
		how the i'th point is ranked in the selection of indices
		for the density filtration. Each point is ranked by the
		distance of its k'th nearest neighbor as per described in
		the Carlsson paper (the smaller the k'th smallest distance,
		the higher it ranks)
	p: 	a float between 0 and 1. The returned indices will be a
		list of length floor(p*n), where n is the number of rows
		in distances_per_point


	Returns the indices that belong to X(k, p)
	"""
	n = distances_per_point.shape[0]
	return_length = int(np.floor(p*n))

	# Get score per row, where the score is the distance from the k'th
	# nearest neighbor
	scores = distances_per_point[:, k-1]

	# sort indices by score
	indices = list(range(n))
	indices.sort(key=lambda x: scores[x])

	return indices[:(return_length+1)]
