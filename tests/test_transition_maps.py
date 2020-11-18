import numpy as np
from manifold_utils.transition_maps import *

def test_iterated_projections():
	c = np.sqrt(2)/2
	p_i = np.array([1, 0])
	p_j = np.array([-c, c])
	L_i = np.array([[1], [0]])
	L_j = np.array([[-c], [c]])

	elbow = iterated_projections(p_i, L_i, p_j, L_j)
	try:
		assert np.all(np.isclose(np.zeros(2), elbow))
	except AssertionError:
		raise AssertionError("Desired outcome was the zero vector; we got the following:\n" + str(elbow))

	length = len_shortest_path(p_i, L_i, p_j, L_j)
	try:
		assert np.all(np.isclose(length, 2.0))
	except AssertionError:
		raise AssertionError("Desired outcome was 2.0; we got "+str(length)+".")

	p_i += np.array([0, 1])
	p_j += np.array([0, 1])

	elbow = iterated_projections(p_i, L_i, p_j, L_j)
	try:
		assert np.all(np.isclose(np.array([0, 1]), elbow))
	except AssertionError:
		raise AssertionError("Desired outcome was (0, 1); we got the following:\n" + str(elbow))

	length = len_shortest_path(p_i, L_i, p_j, L_j)
	try:
		assert np.all(np.isclose(length, 2.0))
	except AssertionError:
		raise AssertionError("Desired outcome was 2.0; we got "+str(length)+".")

	p_i = np.zeros(2)
	p_j = np.array([0, 1])
	elbow = iterated_projections(p_i, L_i, p_j, L_i)
	try:
		assert np.isnan(elbow)
	except:
		raise AssertionError("elbow was the following:\n" + str(elbow))

def test_chart_to_ambient_space():
	"""
	We look at the scenario where our point cloud
	is sampled from a two-dimensional submanifold
	of R^4. We assume that about some point p, the
	principle curvatures of the quadratic
	approximation are all 1.0. We further assume
	that the tangent plane at p is parallel to the
	xy-plane.
	"""
	p = np.array([1, 2, 3, 4])	# center point of approximation
	K = np.ones((2, 2))	#matrix of principle curvatures
	# Columnspace of V is xy-plane
	V = np.array([[1, 0], [0, 1], [0, 0], [0, 0]])
	# Columnspace of V is zw-plane
	V_perp = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])

	# The point x = [1, 1] in local coordinates should be mapped
	# to [2, 3, 4, 5] in the ambient space. Similarly, the point
	# x = [-1, -1] in local coordinate should be mapped to
	# [0, 1, 4, 5] in the ambient space.
	x = np.array([[1, 1], [-1, -1]]).T
	amb_val = chart_to_ambient_space(x, p, V, V_perp, K)
	assert np.all(np.isclose(amb_val, np.array([[2, 3, 4, 5], [0, 1, 4, 5]]).T))
