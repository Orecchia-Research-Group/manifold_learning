from manifold_utils.mSVD import hypersphere
from combined_mls_pca import mls_pca
import numpy as np
import random
import math


## Create S^2 Embedded in R^3
hs=hypersphere(1000,2)
hs=np.stack(hs, axis=1)
zeros_mat=np.zeros((1000,1))
new_hs=np.append(hs,zeros_mat,axis=1)
hs_noisy=new_hs+np.random.randn(1000,3)*math.sqrt(.1)
dim_mat=np.shape(hs_noisy)


uncov_hash=set(map(tuple,hs_noisy))
lla=[]


while (len(uncov_hash) != 0):
    len_uncov=len(uncov_hash)
    print(len_uncov)
    center_tuple=random.sample(uncov_hash,1)
    center_list=np.array([i for i in center_tuple[0]])
    center_ind=np.where(hs_noisy==center_list)[0][0]
    eigvals,eigvecs,radii,Rmin,Rmax,points = mls_pca(hs_noisy,center_ind,2)
    lla_app=[eigvals,center_list,Rmax]  #need to figure out how to extract the center point when there $
    lla.append(lla_app)

    points_hash=set(map(tuple,points))
    uncov_hash = uncov_hash - points_hash


    #for i in points:
        #for j in uncovered_list:
            #boolean = points[i,:] == uncovered_list[j,:]
            #if np.allclose(i,j):
                #np.delete(uncovered_list,uncovered_list.index(j),0)

print(lla)


