from manifold_utils.mSVD import hypersphere
from combined_mls_pca import mls_pca
import numpy as np
import random
import math


## Create S^9 Embedded in R^100
hs=hypersphere(1000,10)
hs=np.stack(hs, axis=1)
zeros_mat=np.zeros((1000,90))
new_hs=np.append(hs,zeros_mat,axis=1)
hs_noisy=new_hs+np.random.randn(1000,100)*math.sqrt(.1)
dim_mat=np.shape(hs_noisy)


uncov_hash=set(map(tuple,hs_noisy))
lla=[]


while (len(uncov_hash) != 0):
    len_uncov=len(uncov_hash)
    print(len_uncov)
    center_tuple=random.sample(uncov_hash,1)
    center_list=np.array([i for i in center_tuple[0]])
    center_ind=np.where(hs_noisy==center_list)[0][0]
    eigvals,eigvecs,radii,Rmin,Rmax,points = mls_pca(hs_noisy,center_ind,9)
    lla_app=[eigvals,center_list,Rmax]  #need to figure out how to extract the center point when there are no indices in sets
    lla.append(lla_app)

    points_hash=set(map(tuple,points))
    uncov_hash = uncov_hash - points_hash
    
print(lla)
