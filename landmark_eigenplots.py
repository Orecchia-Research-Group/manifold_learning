from time import time
import numpy as np
import matplotlib.pyplot as plt
import ilc_data.ilc_loader as ILC
from manifold_utils.mSVD import eigen_calc_from_dist_mat, eigen_plot

data = ILC.get_sct_var_scale()
dist_mat = ILC.get_dist_mat()

landmark_indices = np.load("data/sample_landmark_indices.npy")
Rend = np.max(dist_mat)

start = time()
for ind in landmark_indices:
	ind_str = str(ind)
	radii, eigval_list, _, _ = eigen_calc_from_dist_mat(data, dist_mat, ind, radint=0.5, Rend=40)
	np.save("data/manual_labeling/radii_"+ind_str+".npy", radii)
	np.save("data/manual_labeling/eigvals_"+ind_str+".npy", eigval_list)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(radii, eigval_list)
	ax.set_ylim(top=100, bottom=0)
	fig.savefig("data/manual_labeling/eigenplot_"+ind_str+".jpg")
	plt.close(fig)

	del(radii)
	del(eigval_list)

print("Time elapsed: "+str(time() - start))
