from triangulation import AugmentedDelaunay
import numpy as np

def test_delaunay_formation():
	points = np.load("data/delaunay_test.npy")

	# Assert failure for lack of point names
	try:
		temp = AugmentedDelaunay(points)
		raise AssertionError
	except TypeError:
		pass

	temp = AugmentedDelaunay(points, list(range(points.shape[0])))
