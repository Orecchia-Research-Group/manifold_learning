import numpy as np
from tqdm import tqdm

class tree_node:
	"""
	This is a great docstring
	"""
	def __init__(self, toople):
		self.name = toople
		self.left_child = None
		self.right_child = None
		self.parent = None
		self.n_descendants = 0

	def __str__(self):
		return str(self.name)

	def __disp__(self):
		return str(self.name)

	def __hash__(self):
		return hash(self.name)

	def __eq__(self, other):
		return self.name == other.name

	def __lt__(self, other):
		if self == other:
			return False
		else:
			self_len = len(self.name)
			other_len = len(other.name)
			if self_len < other_len:
				max_ind = self_len
				default_return = True
			else:
				max_ind = other_len
				default_return = False
			ind = 0
			while True:
				if ind == max_ind:
					return default_return
				elif self.name[ind] < other.name[ind]:
					return True
				else:
					ind += 1

	def get_root(self):
		if self.parent is None:
			return self
		else:
			return self.parent.get_root()

class dendro_forest:
	"""
	Attributes:
	--n_leaves (int): indices of leaf nodes go from 0 to n_leaves-1
	--root_map (dict): maps integers to their leaf nodes
	--merges (list): a list of 2-tuples, where each tuple is a
		(float, tree_node) pair. The float denotes the time of
		the merge event creating the tree_node instance.
	"""
	def __init__(self, n):
		self.n_leaves = n
		self.root_map = dict()
		for j in range(n):
			lone_node = tree_node((j,))
			self.root_map[j] = lone_node
		self.merges = []

	def merge(self, a, b, dist):
		"""
		a, b are integers, and are used as node indices
		"""
		# rep_a is the root of the tree containing leaf a.
		# rep_b is defined similarly.
		rep_a = self.root_map[a]
		rep_b = self.root_map[b]

		# Only recognize merge if rep_a, rep_b are distinct
		if rep_a != rep_b:
			# Define left and right children of new node
			if rep_a < rep_b:
				left_rep = rep_a
				right_rep = rep_b
			else:
				left_rep = rep_b
				right_rep = rep_a

			# Create name of new node
			new_name = list(left_rep.name + right_rep.name)
			new_name.sort()
			new_name = tuple(new_name)

			# Create new node, and create parent-child
			# connections
			merge_node = tree_node(new_name)
			merge_node.left_child = left_rep
			merge_node.right_child = right_rep
			left_rep.parent = merge_node
			right_rep.parent = merge_node

			# assign n_descendants to merge_node
			merge_node.n_descendants = left_rep.n_descendants + right_rep.n_descendants + 2

			# Update root_map
			for ind in new_name:
				self.root_map[ind] = merge_node

			# update merges
			self.merges.append((dist, merge_node))

def dendrograph(dist_mat, max_dist=np.inf):
	"""
	Takes a pairwise distance matrix and returns a dendro_forest
	instance encoding all of the merge events in the formation of
	the dendrogram.

	If a positive float max_dist is passed, then the dendrogram is
	formed only up to merges corresponding to distances less than
	or equal to max_dist

	A NumPy array of shape (n, n) results in a dendro_forst instance
	with n leaf nodes
	"""
	# Initialize dendrograph
	n = dist_mat.shape[0]
	forest = dendro_forest(n)

	# create all potential merge events
	merge_candidates = []
	for j in range(n):
		for k in range(j+1, n):
			cand_dist = dist_mat[j, k]
			if cand_dist <= max_dist:
				merge_candidates.append((j, k, cand_dist))

	# sort potential merge events by event distance
	merge_candidates.sort(key=lambda x: x[2])

	# process candidate merge events
	for a, b, dist in tqdm(merge_candidates):
		forest.merge(a, b, dist)

	return forest
