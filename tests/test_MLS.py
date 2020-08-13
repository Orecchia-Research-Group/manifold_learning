import numpy as np
from mls.moving_least_squares import weight, weight_scaled, dweight, dweight_scaled, ddweight, ddweight_scaled, C_1_MLS_oracle

def test_MLS():
    # Creating arrays of points to test
    a = np.array([-2,-1,0,1,2])
    b = np.array([0,3,4,3,0])
    points = np.array([a,b])

    # Delta is set to 5 (to exclude one point), m is set to 2
    MLS = C_1_MLS_oracle(points, 5, 2)

    #Testing whether approximation p*(x) equals value calculated by hand
    assert np.isclose(-5, MLS.eval(3)[0])

    assert np.isclose(-5, MLS.eval(-3)[0])

def test2_MLS():
    # Creating arrays of points
    a = np.array([-5,0,2,3])
    b = np.array([-1,4,2,-1])
    points = np.array([a,b])

    # Delta is set to 8 (to exclude one point), m is set to 2
    MLS = C_1_MLS_oracle(points, 8, 2)

    assert np.isclose(-18, MLS.eval(6)[0])

    # Delta is set to 10 (to exclude one point), m is set to 2
    MLS = C_1_MLS_oracle(points, 10, 2)

    assert np.isclose(-7,MLS.eval(-7)[0])

def test3_MLS():
    # Creating arrays of points
    a = np.array([-20,-5,-3,1])
    b = np.array([16,8,9,1])
    points = np.array([a,b])

     # Delta is set to 8 (to exclude one point), m is set to 2
    MLS = C_1_MLS_oracle(points, 8, 2)

    assert np.isclose(8.91666667, MLS.eval(-4)[0])


def test_MLS_slope():
     # Creating arrays of points
    a = np.array([0,1,2,3])
    b = np.array([0,1,4,9])
    points = np.array([a,b])

    # Delta is set to 5 (to include all points), m is set to 2
    MLS = C_1_MLS_oracle(points, 5, 2)

    #Testing whether approximation of derivative p*'(x) equals value calculated by hand
    assert np.isclose(10, MLS.eval(5)[1])

def test_MLS_slope2():
     # Creating arrays of points to test
    a = np.array([-2,-1,0,1,2])
    b = np.array([0,3,4,3,0])
    points = np.array([a,b])

    # Delta is set to 5 (to exclude one point), m is set to 2
    MLS = C_1_MLS_oracle(points, 5, 2)

    assert np.isclose(6, MLS.eval(-3)[1])

# Testing insert method

def test_insert():
    # Creating arrays of points to test
    a = np.array([-2,-1,0,1,2])
    b = np.array([0,3,4,3,0])
    points = np.array([a,b])

    MLS = C_1_MLS_oracle(points, 5, 2)

    a =  np.array([-2,-1,0])
    b = np.array([0,3,4])
    points = np.array([a,b])

    MLS2 = C_1_MLS_oracle(points, 5, 2)

    a =  np.array([1,2])
    b = np.array([3,0])
    new = np.array([a,b])
    MLS2.insert(new)

    assert np.isclose(MLS.eval(-3)[0], MLS2.eval(-3)[0])
    assert np.isclose(MLS.eval(-3)[1], MLS2.eval(-3)[1])

# Testing insert method again

def test_insert_2():
    # Creating arrays of points to test
    a = np.array([-5,-3,1,5,8])
    b = np.array([-4,5,6,9,10])
    points = np.array([a,b])

    MLS = C_1_MLS_oracle(points, 50, 2)

    a =  np.array([-5,-3,1])
    b = np.array([-4,5,6])
    points = np.array([a,b])

    MLS2 = C_1_MLS_oracle(points, 50, 2)

    a =  np.array([5,8])
    b = np.array([9,10])
    new = np.array([a,b])
    MLS2.insert(new)

    assert np.isclose(MLS.eval(4)[0], MLS2.eval(4)[0])
    assert np.isclose(MLS.eval(4)[1], MLS2.eval(4)[1])
