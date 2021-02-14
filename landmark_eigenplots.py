from time import time
import numpy as np
import matplotlib.pyplot as plt
import ilc_data.ilc_loader as ILC
from manifold_utils.mSVD import rapid_eigen_calc_from_dist_mat, eigen_plot

data = ILC.get_sct_var_scale()
dist_mat = ILC.get_dist_mat()

landmark_indices = np.load("data/sample_landmark_indices.npy")
Rend = np.max(dist_mat)

start = time()
for ind in landmark_indices:
	ind_str = str(ind)
	radii, eigval_list, eigvec_list = rapid_eigen_calc_from_dist_mat(data, dist_mat, ind, radint=1, k=5)
	np.save("data/manual_labeling/radii_"+ind_str+".npy", radii)
	np.save("data/manual_labeling/eigvals_"+ind_str+".npy", eigval_list)
	np.save("data/manual_labeling/eigvecs_"+ind_str+".npy", eigvec_list)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(radii, eigval_list, "b")
	fig.savefig("data/manual_labeling/eigenplot_"+ind_str+".pdf")
	plt.close(fig)
