from triangulation import power_set_bar_empty, all_faces_from_d_simplices, AugmentedDelaunay
import numpy as np

def test_power_set_bar_empty():
	test_tuple = (1, 2, 3, 4)
	test_output = power_set_bar_empty(test_tuple)
	try:
		assert test_output == [(1,), (2,), (3,), (4,), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4), (1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4), (1, 2, 3, 4)]
	except AssertionError:
		raise ValueError

def test_all_faces_from_d_simplices():
	# tetrahedron
	tetra_A = (1, 2, 3, 4)
	test_simplices = [tetra_A]
	output = all_faces_from_d_simplices(test_simplices)
	### collect faces of this tetrahedron
	desired_faces_a = set()
	##### 0-faces
	desired_faces_a.add((1,))
	desired_faces_a.add((2,))
	desired_faces_a.add((3,))
	desired_faces_a.add((4,))
	##### 1-faces
	desired_faces_a.add((1, 2))
	desired_faces_a.add((1, 3))
	desired_faces_a.add((1, 4))
	desired_faces_a.add((2, 3))
	desired_faces_a.add((2, 4))
	desired_faces_a.add((3, 4))
	##### 2-faces
	desired_faces_a.add((1, 2, 3))
	desired_faces_a.add((1, 2, 4))
	desired_faces_a.add((1, 3, 4))
	desired_faces_a.add((2, 3, 4))
	##### 3-face
	desired_faces_a.add(tetra_A)

	assert output == desired_faces_a

	# 2 tetrahedra sharing a single 2-face
	tetra_B = (0, 1, 2, 3)
	test_simplices = [tetra_A, tetra_B]
	output = all_faces_from_d_simplices(test_simplices)
	### collect faces of this simplical complex
	desired_faces_b = desired_faces_a.copy()
	##### 0-faces
	desired_faces_b.add((0,))
	##### 1-faces
	for j in range(1, 4):
		desired_faces_b.add((0, j))
	##### 2-faces
	desired_faces_b.add((0, 1, 2))
	desired_faces_b.add((0, 1, 3))
	desired_faces_b.add((0, 2, 3))
	##### 3-faces
	desired_faces_b.add(tetra_B)

	try:
		assert output == desired_faces_b
	except AssertionError:
		raise ValueError("\n" + ", ".join(str(item) for item in output) + "\n\n" + ", ".join(str(item) for item in desired_faces_b))

	# 2 tetrahedra sharing a single edge
	tetra_C = (1, 2, 5, 6)
	test_simplices = [tetra_A, tetra_C]
	output = all_faces_from_d_simplices(test_simplices)
	### collect faces of this simplicial complex
	desired_faces_c = desired_faces_a.copy()
	##### 0-faces
	desired_faces_c.add((5,))
	desired_faces_c.add((6,))
	##### 1-faces
	desired_faces_c.add((1, 5))
	desired_faces_c.add((1, 6))
	desired_faces_c.add((2, 5))
	desired_faces_c.add((2, 6))
	desired_faces_c.add((5, 6))
	##### 2-faces
	for j in range(4):
		desired_faces_c.add(tuple(tetra_C[k] for k in range(4) if k != j))
	##### 3-faces
	desired_faces_c.add(tetra_C)

	assert output == desired_faces_c

	# 2 tetrahedra sharing a single vertex
	tetra_D = (4, 5, 6, 7)
	test_simplices = [tetra_A, tetra_D]
	output = all_faces_from_d_simplices(test_simplices)
	### collection faces of this simplicial complex
	desired_faces_d = desired_faces_a.copy()
	##### 0-faces
	for j in range(5, 8):
		desired_faces_d.add((j,))
	##### 1-faces
	desired_faces_d.add((4, 5))
	desired_faces_d.add((4, 6))
	desired_faces_d.add((4, 7))
	desired_faces_d.add((5, 6))
	desired_faces_d.add((5, 7))
	desired_faces_d.add((6, 7))
	##### 2-faces
	for j in range(4, 8):
		desired_faces_d.add(tuple(k for k in range(4, 8) if k != j))
	##### 3-faces
	desired_faces_d.add(tetra_D)

	assert output == desired_faces_d

def test_delaunay_formation():
	points = np.load("data/delaunay_test.npy")

	# Assert failure for lack of point names
	try:
		temp = AugmentedDelaunay(points)
		raise AssertionError
	except TypeError:
		pass

	temp = AugmentedDelaunay(points, list(range(points.shape[0])))

