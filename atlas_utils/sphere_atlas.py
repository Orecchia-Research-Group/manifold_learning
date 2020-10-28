import itertools as it
import numpy as np

class coor_chart:
	def __init__(self, name):
		assert isinstance(name, str)
		self.name = name

class atlas_representation:
	def __init__(self):
		pass

def lattice_in_unit_ball(dim, n_points_in_orthant):
	assert isinstance(dim, int)
	assert isinstance(n_points_in_orthant, int)
	delta = 1 / n_points_in_orthant
	one_dim = np.arange(-1, 1, delta).tolist()
	points = set()
	candidates = it.product(one_dim, repeat=dim)
	for candidate in candidates:
		sqrs = [x**2 for x in candidate]
		if np.sum(sqrs) < 1:
			points.add(candidate)
	return points

def unit_ball_to_positive_hemisphere(x, dim):
	y = np.zeros(dim + 1)
	for j in range(dim):
		y[j] = x[j]
	y[dim] = 1 - np.sum(x**2)
	return y

def unit_ball_to_negative_hemisphere(x, dim):
	y = unit_ball_to_positive_hemisphere(x, dim)
	y[dim] = -y[dim]
	return y

# permutation matrix defined for parametrize_chart
def permutation_matrix(dim):
	P = np.zeros((dim, dim))
	for j in range(dim):
		P[j, (j-1) % dim] = 1
	return P

def parametrize_chart(x, P):
	# For specific case of points in S^{n-1}
	assert np.isclose(np.linalg.norm(x), 1)
	ortho = np.array([x])
	pre_para = []
	for j in range(dim - 1):
		if j == 0:
			pre_para.append(P @ x)
		else:
			pre_para.append(P @ x[-1])
	para = np.stack(pre_para, axis=0)
	# kappa is identical for all unit vectors in tangent plane
	kappa = 1

goober = lattice_in_unit_ball(2, 3)
