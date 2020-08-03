import numpy as np
from mls.moving_least_squares import weight, weight_scaled, dweight, dweight_scaled, ddweight, ddweight_scaled, C_1_MLS_oracle

def test_MLS():
    # Creating arrays of points
    a = np.array([-2,-1,0,1,2])
    b = np.array([0,3,4,3,0])
    points = np.array([a,b])

    # Delta is set to 5 (to exclude one point), m is set to 2
    MLS = C_1_MLS_oracle(points, 5, 2)

    assert np.isclose(-5, MLS.eval(3))
    assert np.isclose(-5, MLS.eval(-3))


