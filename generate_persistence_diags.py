import sys
from time import time
import numpy as np
from tqdm import tqdm
from ripser import ripser
from covariance_mat_gen import *

start = time()

# Fix random seed
np.random.seed(42)

# Generate N covariance matrices of size d x d and
# save them as a .npy file
###ds = list(range(5, 31))
d = int(sys.argv[1])
N = 10000

"""
for d in ds:
	print("d = "+str(d)+":")
	Sigmas = []
	for _ in tqdm(range(N)):
		Sigma = random_covariance_matrix(d)
		Sigmas.append(Sigma)

	# Export .npy file
	np.save("data/covariance_mats_"+str(d)+"_"+str(N)+".npy",
			Sigmas)
"""
Sigmas = []
for _ in range(N):
	Sigma = random_covariance_matrix(d)
	Sigmas.append(Sigma)

	np.save("/scratch/robinett/covariance_mats_"+str(d)+"_"+str(N)+".npy",
			Sigmas)

del Sigma
del Sigmas

n = 5000

diags = []
for Sigma in Sigmas:
	X = sample_from_gaussian(n, Sigma)
	dgms = ripser(X)["dgms"]
	diags.append(dgms)

np.save("/scratch/robinett/pers_diags_"+str(d)+"_"+str(N)+".npy",
	diags)
