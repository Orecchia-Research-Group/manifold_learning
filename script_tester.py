from manifold_utils.manifold_sampling import sample_grassmann as sg
from manifold_utils.mSVD import eigen_calc as ec
from manifold_utils.mSVD import eigen_plot as ep
from manifold_utils.mSVD import eps_projection
from manifold_utils.mSVD import hypersphere
import numpy as np
import random
import math


## Recreate figure 1
hs=hypersphere(1000,10)
hs=np.stack(hs, axis=1)
#print(hs)
#print(np.shape(hs))
zeros_mat=np.zeros((1000,90))
#print(np.shape(zeros_mat))
new_hs=np.append(hs,zeros_mat,axis=1)
#print(new_hs)
#print(np.shape(new_hs))
hs_noisy=new_hs+np.random.randn(1000,100)*math.sqrt(.1)
#print(hs_noisy)
dim_mat=np.shape(hs_noisy)
rng=random.randint(0,dim_mat[0])
center=hs_noisy[rng,:]
eigval_list,top_eigvecs,X_mat = ec(hs_noisy,center,9,1.1,2.2,.01)
ep(eigval_list,1.1,2.2,.01)



## Obtain and reshape  cloud point matrix (N x D) from sample_grassmann function
#cloud=sg(4,3,1000)
#cloud_mat=np.reshape(cloud,(1000,12))
#print(np.shape(cloud))
#print(np.shape(cloud_mat))
#print(cloud_mat)

## Pick a random point to act as the center of the sphere
#dim_mat=np.shape(cloud_mat)
#rng=random.randint(0,dim_mat[0])
#center=cloud_mat[rng,:]

## Feed cloud_mat and center into the eigen_calc function
#eigval_list,top_eigvecs,X_mat=ec(cloud_mat,center,2,.1,1,.01)
#print(np.array(eigval_list))
#print(np.shape(eigval_list))
#print(np.array(top_eigvecs))
#print(np.shape(top_eigvecs))
#print(X_mat)
#print(np.shape(X_mat))

#In X_mat, we are grabbing the points from the last radius value, therefore we need to grab the eigenvectors from the last radius value as well.
#print(np.shape(top_eigvecs[90]))

##Plug into projection function
#eigvecs=top_eigvecs[90]
#projection=eps_projection(X_mat,eigvecs,center,2)
#print(projection)

##Plot
#ep(eigval_list,.1,1,.01)



