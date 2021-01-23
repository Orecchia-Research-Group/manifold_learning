import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# read in landmark indices
indices = np.load("data/natural_images/indices.npy")
indices = indices.astype(int)

# Read in images and convert to grayscale vectors
im_vecs = []
for ind in tqdm(indices):
	img = Image.open("figures/natural_images/autocorr_bleach_"+str(ind)+".png").convert("L")
	mat = np.array(img)
	vec = np.reshape(mat, (480*640,))
	im_vecs.append(vec)
im_mat = np.stack(im_vecs, axis=0)

# Perform PCA in 2 dims
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
print("Performing PCA...")
pca.fit(im_mat)
print("Done")
low_dim = pca.transform(im_mat)

# Plot clusters
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(low_dim[:, 0], low_dim[:, 1])

ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")

plt.show()

# Make histogram of clusters
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(low_dim[:, 0], bins=10)

plt.show()
