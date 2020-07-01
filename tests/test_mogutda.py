import numpy as np
from scipy.spatial import Delaunay
from mogutda import SimplicialComplex as sc

def test_ball_betti_comp():
        for j in range(2, 10):
        	simplex = tuple(range(j))
	        complex = sc(simplices=[simplex])
	        for k in range(j):
                        if k == 0:
                                assert complex.betti_number(0) == 1
                        else:
                                assert complex.betti_number(k) == 0

def test_delaunay():
	points = np.load("data/delaunay_test.npy")
	delaunay = Delaunay(points)
	complex = sc(simplices=delaunay.simplices)
	for j in range(4):
		if j == 0:
			assert complex.betti_number(j) == 1
		else:
			assert complex.betti_number(j) == 0
