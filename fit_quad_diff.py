import numpy as np
from sklearn import linear_model 
from manifold_utils.mSVD import eps_projection

def manifold_regression(V,V_perp,points,center):
    #Set up yi and xi to be passed into linear regression function
    k=np.shape(points)[1]
    xi=np.stack(eps_projection(points,eigvecs,center,k))  
    yi=np.stack(eps_projection(points,V_perp,center,k)) 
    xi_squared=np.sqaure(xi)

    #Pass variables into Lin Reg function
    lreg=linear_model.LinearRegression()
    lreg.fit(xi_squared,yi)

    return(lreg.coef_)
    #print xi_sqaured,yi

    
    
    
