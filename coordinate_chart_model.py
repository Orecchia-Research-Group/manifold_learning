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


uncovered_list=hs_noisy
lla=[]


while (len(uncovered_list) != 0):
    dim_uncovered=np.shape(uncovered_list)
    print(dim_uncovered)
    center=random.randint(0,dim_uncovered[0])
    print(center)
    eigvals,eigvecs,radii,Rmin,Rmax,points = mls_pca(hs_noisy,center,9)
    lla_app=[eigvals,uncovered_list[center,:],Rmax]
    lla.append(lla_app)

    for i in range(len(points)):
        for j in range(len(uncovered_list)):
            #boolean = points[i,:] == uncovered_list[j,:]
            if np.allclose(points[i,:],uncovered_list[j,:]) == True:
                np.delete(uncovered_list,j,0)

print(lla)
