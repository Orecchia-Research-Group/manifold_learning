import numpy as np
import matplotlib.pyplot as plt

radii_1 = np.load("scale_lite/radii_0.npy")
eigvals_1 = np.load("scale_lite/eigvals_0.npy")
eigvecs_1 = np.load("scale_lite/eigvecs+0.npy")

radii_2 = np.load("scale_intermediates/radii_0.npy")
eigvals_2 = np.load("scale_intermediates/eigvals_0.npy")
eigvecs_2 = np.load("scale_intermediates/eigvecs+0.npy")

fig = plt.figure(figsize=(5, 10))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.plot(radii_1, eigvals_1)
ax2.plot(radii_2, eigvals_2)

ax1.set_title("n_iter = 1")
ax2.set_title("n_iter = 2")

plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
jordan_coefs = []
for vecs_1, vecs_2 in zip(eigvecs_1, eigvecs_2):
	_, s, _ = np.linalg.svd(vecs_1.T @ vecs_2)
	jordan_coefs.append(s)

ax.plot(radii_1, jordan_coefs)
ax.set_xlabel("radius")
ax.set_ylabel("Inner Product SVs")

plt.show()
