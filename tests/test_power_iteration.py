import numpy as np
from manifold_utils.power_iteration import *

cos45 = np.sqrt(2)/2

def test_power_iteration():
	"""
	See that power_iteration and iterated_power_method succeed in
	eigendecomposition on matrix with eigenvalues 3, 2, 1 and
	eigenvectors (cos45, cos45, 0), (cos45, -cos45, 0), (0, 0, 1)
	"""
	V = np.array([[cos45, cos45, 0], [cos45, -cos45, 0], [0, 0, 1]])
	Lambda = np.diag([3, 2, 1])
	A = V @ Lambda @ V.T

	eigvals, eigvecs = iterated_power_method(A, 3)

	# Check eigenvalues
	try:
		assert np.isclose(eigvals[0], 3, rtol=1e-3)
	except AssertionError:
		raise AssertionError("Top eigenvalue should be 3; got "+str(eigvals[0]))
	try:
		assert np.isclose(eigvals[1], 2, rtol=1e-3)
	except AssertionError:
		raise AssertionError("Middle eigenvalue should be 2; got "+str(eigvals[1]))
	try:
		assert np.isclose(eigvals[2], 1, rtol=1e-3)
	except AssertionError:
		raise AssertionError("Bottom eigenvalue should be 1; got "+str(eigvals[2]))

	# Check eigenvectors
	try:
		assert np.isclose(np.abs(np.dot(eigvecs[0], np.array([cos45, cos45, 0]))), 1)
	except AssertionError:
		raise AssertionError("Top eigenvector should be (cos(45), cos(45), 0); got "+str(eigvecs[0]))
	try:
		assert np.isclose(np.abs(np.dot(eigvecs[1], np.array([cos45, -cos45, 0]))), 1)
	except AssertionError:
		raise AssertionError("Top eigenvector should be (cos(45), -cos(45), 0); got "+str(eigvecs[0]))
	try:
		assert np.isclose(np.abs(np.dot(eigvecs[2], np.array([0, 0, 1]))), 1)
	except AssertionError:
		raise AssertionError("Top eigenvector should be (0, 0, 1); got "+str(eigvecs[2]))
