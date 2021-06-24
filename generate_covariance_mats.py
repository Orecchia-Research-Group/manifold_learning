from time import time
import numpy as np
from tqdm import tqdm
from covariance_mat_gen import *

start = time()

# Fix random seed
np.random.seed(42)

# Generate N covariance matrices of size d x d and
# save them as a .npy file
ds = list(range(5, 31))
N = 100000

for d in ds:
	print("d = "+str(d)+":")
	Sigmas = []
	for _ in tqdm(range(N)):
		Sigma = random_covariance_matrix(d)
		Sigmas.append(Sigma)

	# Export .npy file
	np.save("data/covariance_mats_"+str(d)+"_"+str(N)+".npy",
			Sigmas)

print("Time elapsed: "+str(time() - start))
