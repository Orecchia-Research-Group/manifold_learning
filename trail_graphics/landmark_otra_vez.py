from time import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

landmark_indices = np.load("data/sample_landmark_indices.npy")

for ind in tqdm(landmark_indices):
	ind_str = str(ind)
	radii = np.load("data/manual_labeling/radii_"+ind_str+".npy")
	eigval_list = np.load("data/manual_labeling/eigvals_"+ind_str+".npy")

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(radii, eigval_list)
	ax.set_ylim(top=100, bottom=0)
	fig.savefig("data/manual_labeling/eigenplot_smol_"+ind_str+".jpg")
	plt.close(fig)
