import numpy as np
import matplotlib.pyplot as plt

indices = list(range(17))

for ind in indices:
	try:
		radii = np.load("scale_intermediates/radii_"+str(ind)+".npy")
		eigvals = np.load("scale_intermediates/eigvals_"+str(ind)+".npy")

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(radii, eigvals)
		#ax.set_ylim(top=100)

		fig.savefig("eigenplot_scale+"+str(ind)+".jpg")

		plt.close(fig)

	except FileNotFoundError:
		break
