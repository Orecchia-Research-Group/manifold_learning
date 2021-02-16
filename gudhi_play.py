import numpy as np
import gudhi

st = gudhi.SimplexTree()

for j in range(3):
	simplices = np.load("data/sample_weak_witness_"+str(j)+".npy")
	for k in range(simplices.shape[0]):
		st.insert(simplices[k, :])

print(st.num_simplices())
print(st.num_vertices())
