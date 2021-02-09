import numpy as np
import gudhi

def choose_landmarks(points, dist_mat, p):
	"""
	Randomly choose landmark points for a witness complex
	from a matrix of points. The number of landmark points
	chosen is equal to L = floor(p * n), where n is the number
	points. Returns an L x (n-L) distance matrix.

	inputs:
	points:	n x d matrix of points, where n is the number of
		points and d is the dimension of the data
	dist_mat: n x n matrix of pointwise distances
	p:	float between 0 and 1; the number of landmark
		points chosen is equal to floor(p * n)

	returns:
	indices: a list of the indices of chosen landmarks, each
		representing rows of the input matrix
	witness_dist_mat: L x (n-L) distance matrix of pointwise
		distances from each landmark to each potential
		witness
	"""
	n = points.shape[0]
	n_landmarks = int(np.floor(p * n))

	# Keep Boolean array denoting whether or not point i is a
	# landmark
	landmarks = np.zeros(n, dtype=bool)
	indices = np.random.choice(np.arange(n), n_landmarks, replace=False)
	landmarks[indices] = True

	# Slice witness_dist_mat from dist_mat
	keep_distance = np.outer(landmarks, ~landmarks)
	witness_dist_mat = dist_mat[keep_distance]

	return indices, witness_dist_mat

def complex_from_witness_dist_mat(witness_dist_mat, strong=False, max_alpha_square=np.inf):
	"""
	Create a witness complex from a L x (n-L) distance matrix,
	where there are L landmarks and (n-L) potential witnesses.
	Does this through the GUDHI implementation of the witness
	complex.


	inputs:
	witness_dist_mat: L x (n-L) distance matrix, where there are
		L landmakrs and (n-L) potential witnesses.
	strong (optional): a bool defaulting to False. Will create a
		weak witness complex if False, will otherwise create
		a strong witness complex
	max_alpha_square (optional): optional argument to pass into
		gudhi.WitnessComplex/gudhi.StrongWitnessComplex

	returns:
	st: instance of the GUDHI SimplexTree class
	"""
	# Create landmark table
	num_landmarks, num_witnesses = witness_dist_mat.shape
	landmark_table = []
	for j in range(num_witnesses):
		# slice witness_dist_mat by witness
		landmark_dists = witness_dist_mat[:, j]
		landmark_indices = list(range(num_landmarks))
		landmark_indices.sort(key=lambda x: landmark_dists[x])

		# append row of landmark table
		landmark_table.append([(ind, landmark_dists[ind]) for ind in landmark_indices])

	# Choose strong (weak) witness complex if strong is True
	# (False)
	if strong:
		Complex = gudhi.StrongWitnessComplex
	else:
		Complex = gudhi.WitnessComplex

	# Create witness complex
	complex = Complex(landmark_table)

	return complex.create_simplex_tree(max_square_alpha)
