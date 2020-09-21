import numpy as np
from pymanopt.manifolds import Sphere
from tqdm import tqdm

# Sample from S^1
sphere = Sphere(2)
n = 1000
points = []
for _ in range(n):
	points.append(sphere.rand())
points = np.stack(points, axis=0)

# Perturb by Gaussian noise
points += 0.2*np.random.randn(1000, 2)

rot_mat = np.zeros((2, 2))
rot_mat[0, 1] = -1
rot_mat[1, 0] = 1

def make_atlas(ncharts):
	# Initialize things
	theta = 2*np.pi / ncharts
	chart_len = 2 * np.cos(theta)
	angles = np.array([2*np.pi/ncharts*j for j in range(ncharts)])
	vertices = np.stack([np.array([np.cos(angle), np.sin(angle)]) for angle in angles], axis=0)

	# Assign randomly generated points
	rules = []
	assignments = []
	dist_cond = np.sqrt(2 - 2*np.cos(theta))
	for vert in vertices:
		rule = np.array([np.linalg.norm(points[j, :] - vert) <= dist_cond for j in range(1000)])
		rules.append(rule)
		assignments.append(points[rule])

	# perform least-squares regression
	sum_err = []
	for vert, assignment in zip(vertices, assignments):
		translates = assignment - vert
		xs = translates @ (rot_mat @ vert)
		ys = translates @ vert
		sq_xs = xs**2
		# optimal coefficient kappa
		kappa = np.dot(sq_xs, ys) / np.dot(sq_xs, sq_xs)
		diff = 0.5 * kappa * sq_xs
		sum_err.append(np.dot(diff, diff))

	# compute number of points left out
	da_rules = np.stack(rules, axis=0)
	included = np.any(da_rules, axis=0)
	num_included = np.sum([1 if x else 0 for x in included])
	# return average error and point-counts, as well as included
	point_counts = np.sum([assignment.shape[0] for assignment in assignments])
	return np.sum(sum_err), point_counts, num_included

avg_errors = []
point_counts = []
nums_included = []
for j in range(4, 21):
	err, counts, num_included = make_atlas(j)
	point_counts.append(counts)
	avg_errors.append(err / counts)
	nums_included.append(num_included)

# plot stuff
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(list(range(4, 21)), avg_errors)

ax.set_xlabel("# charts")
ax.set_ylabel("average squared error")

fig.savefig("more_chart_is_more_gooder.png")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(list(range(4, 21)), point_counts)

ax.set_xlabel("# charts")
ax.set_ylabel("# point-counts")

fig.savefig("point_counts.png")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(list(range(4, 21)), nums_included)

ax.set_xlabel("# charts")
ax.set_ylabel("# points accounted for")
ax.set_ylim((0, ax.get_ylim()[1]))

fig.savefig("counted.png")
