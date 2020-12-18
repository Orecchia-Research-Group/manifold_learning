import numpy as np
from sklearn import linear_model 
from manifold_utils.mSVD import eps_projection

def manifold_regression(V,V_perp,points,center):
    #Set up yi and xi to be passed into linear regression function
    #k=np.shape(points)
    new_points = points - center
    xi=np.dot(new_points,V)  
    yi=np.dot(new_points,V_perp) 
    xi_squared=np.square(xi)

    #Pass variables into Lin Reg function
    lreg=linear_model.LinearRegression()
    lreg.fit(xi_squared,yi)

    #print(xi.shape)
    #print(yi.shape)
    return(lreg.coef_)

    
    
    
