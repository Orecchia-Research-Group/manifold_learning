import numpy as np
from pymanopt.manifolds import Grassmann
from manifold_utils.iga import chakraborty_express

def test_chakraborty_express():
	# Sample points from 6-11 Grassmann
	Gr = Grassmann(11, 6)
	X = Gr.rand()
	Y = Gr.rand()

	# Assert identity at beginning and end of path
	X_prime = chakraborty_express(X, Y, 0)
	U, s, Vh = np.linalg.svd(X.T @ X_prime)
	try:
		assert np.all(np.isclose(s, np.ones(6)))
	except AssertionError:
		print(str(U)+"\n\n"+str(s)+"\n\n"+str(Vh))
	Y_prime = chakraborty_express(X, Y, np.pi/2)
	_, s, _ = np.linalg.svd(Y.T @ Y_prime)
	try:
		assert np.all(np.isclose(s, np.ones(6)))
	except AssertionError:
		print(str(U)+"\n\n"+str(s)+"\n\n"+str(Vh))
