import numpy as np
from scipy.linalg import qr

def random_orthogonal_matrix(d):
	"""
	Return orthogonal matrix, sampled uniformly from all
	orthogonal matrices of size d x d.
	This is stylized after the random generation of
	orthogonal matrices in pymanopt
	TODO: properly cite

	inputs: d (int): dimension of the orthogonal matrix
		(i.e. returns a d x d matrix)
	outputs: Q (np.ndarray): orthogonal matrix
	"""
	# Generate matrix from random uniform distribution
	A = np.random.rand(d, d)
	# Perform QR decomposition on A
	Q, R = qr(A)
	# Return Q, with approprite sign corrections
	Q = np.dot(Q, np.diag(np.sign(np.diag(R))))

	return Q

def random_pd_diagonal(d):
	"""
	Return diagonal matrix d x d whose entries are a
	vector of unit length from the positive orthant in
	R^d

	Stylized after an SO post I need to properly cite
	TODO: properly cite

	inputs: d (int): dimension of the PD diagonal matrix
		(i.e. returns a d x d matrix)
	outputs: Lam (np.ndarray): positive-definite diagonal
		matrix
	"""
	# Generate vector from random uniform distribution
	v = np.random.rand(d)
	# Rescale to unit-length vector
	lam = v/np.linalg.norm(v)
	# Move lam to positive orthant, and diagonalize
	lam = lam * np.sign(lam)
	Lam = np.diag(lam)

	return Lam

def random_covariance_matrix(d):
	"""
	Return random covariance matrix as formalized in the manuscript
	TODO: properly cite

	inputs: d (int): dimension of the covariance matrix
		(i.e. returns a d x d matrix)
	outputs: Sigma (np.ndarray): covariance matrix
	"""
	# Generate random orthogonal matrix
	P = random_orthogonal_matrix(d)
	# Generate eigenvalues
	Lam = random_pd_diagonal(d)
	# Generate covariance matrix
	Sigma = P @ Lam @ P.T

	return Sigma

def sample_from_gaussian(n, Sigma):
	"""
	Return n x d matrix, where each row is a d-dimensional vector
	sampled from centered normal distribution with covariance
	matrix Sigma

	inputs: n (int): number of points to sample
		Sigma (np.ndarray): d x d covariance matrix

	outputs: X (np.ndarray): n x d matrix of sampled data
	"""
	d = Sigma.shape[0]
	# Initialize zero vector as center of distribution
	mu = np.zeros(d)
	# Empty list for collecting points
	points = []
	# Sample points
	X = np.random.multivariate_normal(mu, Sigma, n)

	return X
