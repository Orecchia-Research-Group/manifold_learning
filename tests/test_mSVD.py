from manifold_utils.mSVD import hypersphere, eigen_calc
import numpy as np

def test_hypersphere():
	"""
	Test that hypersphere genuinely samples points
	uniformly from the surface of the unit hypersphere
	"""
	n = 10000
	for d in [3, 5, 7]:
		points = hypersphere(n, d)
		for j in range(n):
			assert np.isclose(np.linalg.norm(points[:, j]), 1)


def test_eigencalc():
    """
    Test
    """
    
