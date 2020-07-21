from manifold_utils.manifold_sampling import sample_grassmann as sg
from manifold_utils.mSVD import eigen_calc as ec
from manifold_utils.mSVD import eigen_plot as ep
import numpy as np
import random


## Obtain and reshape  cloud point matrix (N x D) from sample_grassmann function
cloud=sg(4,3,1000)
cloud_mat=np.reshape(cloud,(1000,12))
print(np.shape(cloud))
print(np.shape(cloud_mat))
print(cloud_mat)

## Pick a random point to act as the center of the sphere
dim_mat=np.shape(cloud_mat)
rng=random.randint(0,dim_mat[0])
center=cloud_mat[rng,:]

## Feed cloud_mat and center into the eigen_calc function
eigval_list,top_eigvecs=ec(cloud_mat,center,2,.1,1,.01)
print(np.array(eigval_list))
print(np.shape(eigval_list))
print(np.array(top_eigvecs))
print(np.shape(top_eigvecs))


##Plot
#ep(eigval_list,.1,1,.01)



