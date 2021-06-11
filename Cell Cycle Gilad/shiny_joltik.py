import sys
from random import sample
import numpy as np
from ripser import Rips, ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
import bats
from tqdm import tqdm
from cell_cycle_utils import get_umap, birth_lifetime

# Read in index of random genes
j = sys.argv[1]

n = 888
inds = list(range(n))
n_randos = 10
smol_size = 300
L2 = bats.Euclidean()
F2 = bats.F2()

# Load in data
mat = np.load("random_proj_"+str(j)+".npy").astype(np.float64)
with open("random_genes_"+str(j)+".txt", "r") as f:
	genes = f.read().split("\n")[:-1]

# Create 2D UMAP representation
umap_rep = get_umap(mat)

# Perform random subsampling
np.random.seed(42)
smol_inds = sample(inds, smol_size)
full_smol = mat[smol_inds, :]
umap_smol = umap_rep[smol_inds, :]

# Compute Z2 homology in BATS
data = bats.DataSet(bats.Matrix(full_smol))
dist_mat = L2(data, data)
print("CP 1")
f = bats.RipsFiltration(data, L2, np.inf, 2)
print("CP 2")
r = bats.reduce(f, F2)
print("CP 3")
cpx = f.complex()

# get ten longest-living features
h1 = r.persistence_pairs(1)
h1.sort(key=lambda x: x.death() - x.birth(), reverse=True)
top_h1 = h1[:10]

print("CP 4")

# Make figures for each feature
for k in tqdm(range(10)):
	# Plot cycle on umap_smol
	pair = top_h1[k]
	thresh = pair.birth()
	fig = plt.figure(figsize=(10, 5))
	ax1 = fig.add_subplot(121)
	ax1.scatter(umap_smol[:, 0], umap_smol[:, 1], c="k", s=0.1)

	rep = r.representative(pair)
	nzind = rep.nzinds()
	for kk in nzind:
		[i_prime, j_prime] = cpx.get_simplex(1, kk)
		if dist_mat[i_prime, j_prime] <= thresh:
			ax1.plot([umap_smol[i_prime, 0], umap_smol[j_prime, 0]],
				[umap_smol[i_prime, 1], umap_smol[j_prime, 1]], c="r")

	# Plot persistence diagram
	ax2 = fig.add_subplot(122)
	dgms = ripser(full_smol, maxdim=1)["dgms"]
	H1 = dgms[1]
	birth_lifetime(H1, ax2)

	### emphasize H1 feature of interest
	ax2.plot([pair.birth()], [pair.death() - pair.birth()], "ks")

	# save figure
	if k == 0:
		suffix= "st"
	elif k == 1:
		suffix = "nd"
	elif k == 2:
		suffix = "rd"
	else:
		suffix = "th"

	fig.suptitle(", ".join(genes)+",\n"+str(k+1)+suffix+" Largest H1 Feature")
	fig.savefig("rando_caricature_"+str(j)+"_"+str(k)+".png")
	plt.close(fig)
