from time import time
import numpy as np
import gudhi

ks = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000]

for kk in ks:
	st = gudhi.SimplexTree()

	for j in range(3):
		simplices = np.load("data/sample_weak_witness_kis"+str(kk)+"_"+str(j)+".npy")
		for k in range(simplices.shape[0]):
			st.insert(simplices[k, :])

	st.compute_persistence(homology_coeff_field=2)
	start = time()
	print(st.betti_numbers())
	print(time() - start)
