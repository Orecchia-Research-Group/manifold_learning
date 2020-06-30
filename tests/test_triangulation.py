from triangulation import power_set_bar_empty, AugmentedDelaunay
import numpy as np

def test_power_set_bar_empty():
	test_tuple = (1, 2, 3, 4)
	test_output = power_set_bar_empty(test_tuple)
	try:
		assert test_output == [(1,), (2,), (3,), (4,), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4), (1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4), (1, 2, 3, 4)]
	except AssertionError:
		raise ValueError

def test_all_faces_from_d_simplices():
	pass

def test_delaunay_formation():
	points = np.load("data/delaunay_test.npy")

	# Assert failure for lack of point names
	try:
		temp = AugmentedDelaunay(points)
		raise AssertionError
	except TypeError:
		pass

	temp = AugmentedDelaunay(points, list(range(points.shape[0])))
