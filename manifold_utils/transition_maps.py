import numpy as np

def iterated_projections(p_i, L_i, p_j, L_j):
	"""
	Finds the elbow point in the shortest path from p_i to p_j
	that is contained entirely within the union of V_i and V_j,
	where V_k is the columnspace of L_k translated by p_k.

	p_i is the centerpoint of a chart in the ambient space
	p_j is the centerpoint of the second chart
	L_i is the N x d matrix whose columns are the top d
		eigenvectors for the local linear approximation
		at p_i
	L_j is the N x d matrix whose columns are the top d
		eigenvectors for the local linear approximation
		at p_j
	"""
	low_rank_i = L_i @ L_i.T
	low_rank_j = L_j @ L_j.T
	trans_i = p_i - low_rank_i @ p_i
	trans_j = p_j - low_rank_j @ p_j

	prev_marg = np.inf
	p_prev = p_i
	p = p_j
	while not np.all(np.isclose(p_prev, p)):
		#Projection corresponding to p_i, L_i
		p_prev = p
		p = low_rank_i @ p + trans_i
		new_marg = np.linalg.norm(p - p_prev)
		if new_marg >= prev_marg:
			return np.nan
		prev_marg = new_marg

		#Projection corresponding to p_j, L_j
		p_prev = p
		p = low_rank_j @ p + trans_j
		new_marg = np.linalg.norm(p - p_prev)
		if new_marg >= prev_marg:
			return np.nan
		prev_marg = new_marg
	return p

def len_shortest_path(p_i, L_i, p_j, L_j):
	"""
	Finds the shortest path from p_i to p_j that is contained
	entirely within the union of V_i and V_j, where V_k is the
	columnspace of L_k translated by p_k.

	p_i is the centerpoint of a chart in the ambient space
	p_j is the centerpoint of the second chart
	L_i is the N x d matrix whose columns are the top d
		eigenvectors for the local linear approximation
		at p_i
	L_j is the N x d matrix whose columns are the top d
		eigenvectors for the local linear approximation
		at p_j
	"""
	elbow = iterated_projections(p_i, L_i, p_j, L_j)
	if np.any(np.isnan(elbow)):
		return np.nan
	else:
		return np.linalg.norm(p_i - elbow) + np.linalg.norm(p_j - elbow)

def chart_to_ambient_space(x, p, V, V_perp, K):
	"""
	x: representation of point in local coordinate chart
		(d-dimensional NumPy array); or
		(dxk NumPy aray for some k)
	p: center of coordinate chart in ambient space
		(N-dimensional NumPy array)
	V: the N x d Stiefel matrix whose columnspace is
		parallel to the local linear approximation
		used to learn the local quadratic
		approximation at p
		(Nxd NumPy array)
	V_perp: the N x (N-d) Stiefel matrix whose columnspace is
		orthogonal to the local linear approximation
		used to learn the local quadratic
		approximation at p
		(Nx(N-d) NumPy array)
	"""
	x_quad = 0.5 * np.square(x)
	before_trans = V @ x + V_perp @ K @ x_quad
	try:
		return p + before_trans
	except ValueError:
		return np.stack([p] * before_trans.shape[1], axis=0).T + before_trans
