import numpy as np

def mag_post_mul(A, v):
	Av = A.dot(v)
	return v.dot(Av)

def power_iteration(A, norm_diff=1e-6):
	"""
	A single iteration of the power iteration method for finding
	eigendecompositions of matrices. Meant to be used internally
	as a subroutine in  iterated_power_method.

	A: nxn ndarray
	norm_diff: positive float (default 0.01)
	"""
	n, d = A.shape

	v = np.random.rand(d)
	v = v / np.linalg.norm(v)
	ev = mag_post_mul(A, v)

	while True:
		Av = A.dot(v)
		v_new = Av / np.linalg.norm(Av)

		ev_new = mag_post_mul(A, v_new)
		if np.abs(ev - ev_new) < norm_diff:
			break

		v = v_new
		ev = ev_new

	return ev_new, v_new

def iterated_power_method(A, k, eigval_list=None, eigvec_list=None, norm_diff=1e-6):
	"""
	Compute the top k eigenvalues of the matrix A. eigval_list
	and eigvec_list store the eigenvalue/vector pairs in order
	of decreasing absolute value (of eigenvalue). It is
	recommended to avoid manually inputing nonempty lists for
	eigval_list and eigvec_list.

	A: nxn ndarray
	k: nonnegative int
	eigval_list: list or None (default None)
	eigvec_list: list or None (default None)
	norm_diff: positive float (default 0.01)
	"""
	# initialize eigval_list and eigvec_list if first iteration
	if (not eigval_list) or (not eigvec_list):
		eigval_list = []
		eigvec_list = []

	# return collected eigenpairs when k == 0
	if k == 0:
		return eigval_list, eigvec_list

	else:
		# compute the top eigenvalue-eigenvector pair using
		# power_iteration
		ev_new, v_new = power_iteration(A, norm_diff=norm_diff)
		eigval_list.append(ev_new)
		eigvec_list.append(v_new)

		# update A to be orthogonal to acquired eigenpair
		B = A - (ev_new * np.outer(v_new, v_new))

		# recurse on updated A
		return iterated_power_method(B, k-1, eigval_list=eigval_list, eigvec_list=eigvec_list, norm_diff=norm_diff)
