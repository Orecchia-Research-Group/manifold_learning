import itertools as it
import numpy as np

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
	y[dim] = np.sqrt(1 - np.sum(x**2))
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
	dim = x.shape[0]
	ortho = np.array(x)
	pre_para = []
	for j in range(dim - 1):
		if j == 0:
			pre_para.append(P @ x)
		else:
			pre_para.append(P @ pre_para[-1])
	para = np.stack(pre_para, axis=0)

	return para, ortho

class coor_chart():
	def __init__(self, root_point, P):
		assert isinstance(root_point, tuple)
		self.name = root_point
		self.para, self.ortho = parametrize_chart(np.array(root_point), P)

class atlas_representation:
	def __init__(self, dim, n_points_in_orthant):
		# Generate permutation matrix
		self.P = permutation_matrix(dim)

		# Generate centerpoints of coordinate charts
		lattice = lattice_in_unit_ball(dim, n_points_in_orthant)
		upper_hemisphere = set(unit_ball_to_upper_hemisphere(point, dim) for tuple(point) in lattice)
		lower_hemisphere = set(unit_ball_to_lower_hemisphere(point, dim) for tuple(point) in lattice)
		self.root_points = upper_hemisphere.union(lower_hemisphere)
		self.charts = {}
		for root_point in self.root_points:
			self.charts[root_point] = coor_chart(root_point, self.P)
