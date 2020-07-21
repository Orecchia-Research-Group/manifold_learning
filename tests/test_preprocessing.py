from manifold_utils.preprocessing import point_collection
import numpy as np

def test_point_collection():
	points = np.load("data/spherical_toy_data.npy")
	dist_mat = np.load("data/spherical_toy_dist.npy")
	pc = point_collection(points, dist_mat)
	from_zero = pc.points_sorted_from_center(0)
