from random import sample
import numpy as np
import matplotlib.pyplot as plt
from ripser import Rips, ripser
from persim import plot_diagrams
import bats
#import plotly
#import plotly.graph_objects as go
from tqdm import tqdm
from time import time

full_rep = np.load("full_rep.npy").astype(np.float64)
four_dim_rep = np.load("4_dim_rep.npy").astype(np.float64)
umap_rep = np.load("umap_rep.npy").astype(np.float64)

# margins for isolating loop in umap_rep
# left_margin is a constant, for a vertical line
# the right margin is represented by a linear function
# of the form y = ax + b
left_margin = 1.0

right_x = np.linspace(-2.0, 12.5, 100)
right_a = -1.0
right_b = 15
right_y = right_a * right_x + right_b

# remove indices that lay outside margins
n_points = full_rep.shape[0]

cut_inds = []
kept_inds = []
for j in range(n_points):
	if umap_rep[j, 0] < left_margin:
		cut_inds.append(j)
	elif ((right_a * umap_rep[j, 0]) + right_b - umap_rep[j, 1] < 0):
		cut_inds.append(j)
	else:
		kept_inds.append(j)

full_cut = full_rep[cut_inds, :]
full_kept = full_rep[kept_inds, :]
four_dim_cut = four_dim_rep[cut_inds, :]
four_dim_kept = four_dim_rep[kept_inds, :]
umap_cut = umap_rep[cut_inds, :]
umap_kept = umap_rep[kept_inds, :]

smol_size = 300
np.random.seed(42)
smol_inds = sample(kept_inds, smol_size)

full_smol = full_rep[smol_inds, :]
four_dim_smol = four_dim_rep[smol_inds, :]
umap_smol = umap_rep[smol_inds, :]

# Free space
del(full_rep)
del(umap_rep)
del(full_cut)
del(full_kept)
del(umap_cut)
del(umap_kept)
print("excess variables deleted")

# Use BATS to compute Rips FIltration
full_data = bats.DataSet(bats.Matrix(full_smol))
L2 = bats.Euclidean()
dist_mat_full = L2(full_data, full_data)

start = time()
f_full = bats.RipsFiltration(full_data, L2, np.inf, 2)
print(time() - start)

start = time()
r_full = bats.reduce(f_full, bats.F2())
print(time() - start)

# get ten longest-living features
h0_full = r_full.persistence_pairs(0)
h1_full = r_full.persistence_pairs(1)

h1_full.sort(key=lambda x: x.death() - x.birth(), reverse=True)
top_h1_full = h1_full[:10]

for j, pair in tqdm(enumerate(top_h1_full)):
	thresh = pair.birth()
	#fig = go.Figure()
	#fig.add_trace(go.Scatter(x=full_smol[:, 0], y=full_smol[:, 1], mode="markers"))
	fig = plt.figure(figsize=(10, 5))
	ax = fig.add_subplot(121)
	ax.scatter(umap_smol[:, 0], umap_smol[:, 1], c="k")
	edge_x = []
	edge_y = []

	"""
	for k in range(smol_size):
		for kk in range(smol_size):
			if dist_mat_full[k, kk] <= thresh:
				ax.plot([full_smol[k, 0], full_smol[kk, 0]],
					[full_smol[k, 1], full_smol[kk, 1]], c="k", lw=0.5)
	"""
				#edge_x.append([full_smol[k, 0], full_smol[kk, 0]])
				#edge_y.append([full_smol[k, 1], full_smol[kk, 1]])

	#fig.add_trace(go.Scatter(x=edge_x, y=edge_y,
	#			line=dict(width=0.5, color="#888"),
	#			hoverinfo="none", mode="lines"))
#	for xs, ys in zip(edge_x, edge_y):
#		ax.plot(xs, ys, c="k", lw=0.5)

	edge_x = []
	edge_y = []
	r = r_full.representative(pair)
	nzind = r.nzinds()
	cpx = f_full.complex()
	for k in nzind:
		[i_prime, j_prime] = cpx.get_simplex(1, k)
		if dist_mat_full[i_prime, j_prime] <= thresh:
#			edge_x.append([full_smol[i_prime, 0], full_smol[j_prime, 0]])
#			edge_y.append([full_smol[i_prime, 1], full_smol[j_prime, 1]])
			ax.plot([umap_smol[i_prime, 0], umap_smol[j_prime, 0]],
				[umap_smol[i_prime, 1], umap_smol[j_prime, 1]], c="r")
	#fig.add_trace(go.Scatter(
	#		x=edge_x, y=edge_y,
	#		line=dict(width=2, color="red"),
	#		hoverinfo="none", mode="lines"))
	#fig.update_layout()
#	for xs, ys in zip(edge_x, edge_y):
#		ax.plot(xs, ys, c="r", lw=1.0)

	# Make persistence diagram
	ax2 = fig.add_subplot(122)
	dgms = ripser(full_smol)["dgms"]
	plot_diagrams(dgms, show=False, ax=ax2)

	# Emphasize H1 feature of interest
	ax2.plot(pair.birth(), pair.death(), "ks")

	fig.savefig("which_cycle_full_"+str(j)+".png")
	plt.close(fig)
	#fig.write_image("which_cycle_full_"+str(j)+".pdf")
