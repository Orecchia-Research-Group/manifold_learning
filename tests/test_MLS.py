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

    # Creating arrays of points
    a = np.array([-5,0,2,3])
    b = np.array([-1,4,2,-1])
    points = np.array([a,b])

    # Delta is set to 8 (to exclude one point), m is set to 2
    MLS = C_1_MLS_oracle(points, 8, 2)

    assert np.isclose(-18, MLS.eval(6))

    # Delta is set to 10 (to exclude one point), m is set to 2
    MLS = C_1_MLS_oracle(points, 10, 2)
    assert np.isclose(-7,MLS.eval(-7))

    # Creating arrays of points
    a = np.array([-20,-5,-3,1])
    b = np.array([16,8,9,1])
    points = np.array([a,b])

     # Delta is set to 8 (to exclude one point), m is set to 2
    MLS = C_1_MLS_oracle(points, 8, 2)

    assert np.isclose(8.91666667, MLS.eval(-4))

    # Delta is set to 5 (to exclude two points), m is set to 2
    MLS = C_1_MLS_oracle(points, 5, 2)

    assert np.isclose(68.3368, MLS.eval(-7))


