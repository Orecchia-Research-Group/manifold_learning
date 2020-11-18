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
