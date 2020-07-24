from manifold_utils.mSVD import hypersphere, eigen_calc, eps_projection
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


### def test_eigencalc():




def test_eps_projection():
    """
    Test that the function maps to the proper points in k dimensions
    """
    vectors = np.stack([[4,2,5,7,6,4,2],[3,5,6,1,3,2,2]])
    eigvecs = np.stack([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
    mapped_points = np.stack([[4,2,5,7,6,4],[3,5,6,1,3,2]])
    center = [0,0,0,0,0,0,0]
    k=6

    assert eps_projection(vectors,eigvecs,center,k) == np.sum(mapped_points)
