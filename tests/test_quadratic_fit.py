from fit_quad_diff import manifold_regression as mr
import numpy as np
import matplotlib.pyplot as plt

def test_manifold_regression():
    ## Sample within unit circle R^2
    length = np.sqrt(np.random.uniform(0,1,1000))

    theta = np.pi * np.random.uniform(0,2,1000)

    x=length * np.cos(theta)
    y=length * np.sin(theta)
    z=0.5*x**2-3.0*y**2

    V=np.vstack([[1,0],[0,1],[0,0]])
    V_perp=np.vstack([[0],[0],[1]])

    table=np.stack((x,y,z),axis=1)
    coef = mr(V,V_perp,table,[0,0,0])
    test_coef =  np.array([[0.5, -3.]])

    assert coef.all() == test_coef.all()
 
