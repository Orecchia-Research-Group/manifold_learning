import numpy as np

def chakraborty_express(X, Y, t):
	"""
	X and Y are N x d Stiefel matrices (i.e. their columns are
	orthonormal). This function computes the "thin" SVD of
	(1 - X@X.T)@Y@(X.T @ Y)^{-1}. It then uses this information
	to translate along a geodesic of the Grassmannian according
	to eqn 4 of Chakraborty et al. 2017

	For context, see between eqns 4 and 5 from Chakraborty et al.
	2017

	NOTE: It must be the case that X.T @ Y is invertible
	"""
	# Get terms A and B of the matrix product A@Y@B in the
	# doctring
	N, d = X.shape	# X,Y should have same shape
	I = np.eye(N)	# N x N identity matrix
	A = I - (X @ X.T)
	B = np.linalg.inv(X.T @ Y)

	# Compute SVD of matrix product
	P = A @ Y @ B
	U, s, Vh = np.linalg.svd(P)

	# Compute matrix of angle arctangents
	Theta = np.diag(np.arctan(s))

	# Move length t along geodesic from X to Y on Grassmann
	cos_term = X @ Vh.T
	try:
		cos_term_complete = cos_term @ np.cos(Theta*t)
	except ValueError:
		err_str = ""
		err_str += "shape of cos_term: "+str(cos_term.shape)+"\n"
		err_str += "shape of Theta: "+str(Theta.shape)+"\n"
		err_str += "shape of P: "+str(P.shape)+"\n"
		raise ValueError(err_str)
	sin_term_complete = U[:, :d] @ np.sin(Theta*t)
	new_mat = cos_term_complete + sin_term_complete

	# Perform Gram-Schmidt on columns of new_mat
	Q, _ = np.linalg.qr(new_mat)

	return Q

def iga(stiefel_list):
	"""
	Performs IGA from Chakraborty et al. Each matrix in stiefel_list must
	be a Stiefel matrix, and all Stiefel matrices must have same shape.

	This method performs an online approximation of the Grassmann average
	using iterated calls to chakraborty_express
	"""
	for j, Y in enumerate(stiefel_list):
		if j == 0:
			X = Y
		else:
			X = chakraborty_express(X, Y, (j+1)**(-1))
	return X

def arccos_catch_nan_scalar(scalar):
	"""
	Implement numpy.arccos on a scalar value, except catch NaNs and
	convert to zero
	"""
	temp = np.arccos(scalar)
	return 0 if np.isnan(temp) else temp

# Implement numpy.arccos on an array, except catch NaNs and convert
# to zero
arccos_catch_nan = np.vectorize(arccos_catch_nan_scalar)
