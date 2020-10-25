from fit_quad_diff import manifold_regression as mr
import numpy as np
import matplotlib.pyplot as plt

## Sample within unit circle R^2
length = np.sqrt(np.random.uniform(0,1,400))
#print(length)

theta = np.pi * np.random.uniform(0,2,400)
#print(theta)

x=length * np.cos(theta)
y=length * np.sin(theta)
z=np.sqrt(1-x**2-y**2)

#plt.scatter(x,y)
#plt.show()

table=np.stack((x,y,z),axis=1)
cov_X = np.cov(table, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(cov_X)  # computes the eigenvalues and eigenvectors of the covariance matrix

eigvecs=np.vstack(eigvecs)
top_eig = eigvecs[:,:2]
bot_eig = eigvecs[:,-1]

print(top_eig)
print(np.shape(top_eig))
print(bot_eig)

coef = mr(top_eig,bot_eig,table,[0,0,0])

#print(coef)
#print(table)

