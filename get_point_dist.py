import numpy as np
import matplotlib.pyplot as plt
from ilc_data.ilc_loader import get_dist_vec_scale

trail_indices = np.load("data/trail_indices.npy")

for j, ind in enumerate(trail_indices):
	dist_vec = get_dist_vec_scale(ind)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.hist(dist_vec, bins=100)
	ax.set_xlabel("Radius")
	ax.set_ylabel("Number of Points")
	ax.set_title("Point Density -- Cheerio "+str(j))

	fig.savefig("point_densities/density_cheerio_"+str(j)+".pdf")
	plt.close(fig)
