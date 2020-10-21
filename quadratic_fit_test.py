from manifold_utils.mSVD import hypersphere
from combined_mls_pca import mls_pca
from fit_quad_diff import manifold_regression
import numpy as np
import math
import sklearn
import random

## Create S^9 Embedded in R^100
hs=hypersphere(1000,10)
hs=np.stack(hs, axis=1)
zeros_mat=np.zeros((1000,90))
new_hs=np.append(hs,zeros_mat,axis=1)
hs_noisy=new_hs+np.random.randn(1000,100)*math.sqrt(.1)
dim_mat=np.shape(hs_noisy)

## Run mls_pca on it
center_ind=random.randrange(0,dim_mat[0])
eigvals,eigvecs,radii,Rmin,Rmax,points,eig_perp = mls_pca(hs_noisy,center_ind,9)
print(np.shape(eigvecs))
print(np.shape(eig_perp))


## Perform quadratic fitting
#manifold_regression(eigvecs[-1,],eig_perp[-1,],points,hs_noisy[center_ind,])

