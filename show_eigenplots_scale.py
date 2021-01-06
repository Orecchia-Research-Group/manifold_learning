import numpy as np
import matplotlib.pyplot as plt
from ilc_data.ilc_loader import get_dist_vec_scale

indices = list(range(17))
trail_indices = np.load("data/trail_indices.npy")

for ind in indices:
	try:
		radii = np.load("scale_intermediates/radii_"+str(ind)+".npy")
		eigvals = np.load("scale_intermediates/eigvals_"+str(ind)+".npy")

		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		dist_vec = get_dist_vec_scale(trail_indices[ind])
		sorted_vec = np.sort(dist_vec)
		sorted_vec = [val for val in sorted_vec if val >= radii[0]]
		sorted_vec = [val for val in sorted_vec if val <= radii[-1]]
		ax1.plot(radii, eigvals)

		ax2 = ax1.twinx()
		ax2.hist(sorted_vec, bins=100, alpha=0.5)

		ax1.set_title("Cheerio "+str(ind)+" in scale_data space")
		ax1.set_xlabel("radius")
		ax1.set_ylabel("eigenvalue")
		ax2.set_ylabel("number of points")

		fig.savefig("eigenplot_scale+"+str(ind)+".jpg")

		plt.close(fig)

	except FileNotFoundError:
		break
