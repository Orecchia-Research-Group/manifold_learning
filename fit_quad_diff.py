import numpy as np
from sklearn import linear_model 
from manifold_utils.mSVD import eps_projection

def manifold_regression(eigvecs,points,center):
    k=np.shape(points)[1]
    xi=np.stack(eps_projection(points,eigvecs,center,k))
    V_perp= ##  
    yi=np.stack(eps_projection(points,V_perp,center,k)) #Needs review
    xi_squared=np.sqaure(xi)

    print xi_sqaured,yi

    
    
    
