import itertools as it
import numpy as np
from atlas_utils.sphere_atlas import *

def test_unit_ball_to_positive_hemisphere():
	ubtph = unit_ball_to_positive_hemisphere
	for j in range(1, 5):
		zero_target = np.zeros(j+1)
		zero_target[-1] = 1.0
		zero_answer = ubtph(np.zeros(j), j)
		try:
			assert np.all(np.isclose(zero_answer, zero_target))
		except AssertionError:
			raise AssertionError("zero_answer: "+str(zero_answer))
		one_target = np.zeros(j+1)
		one_target[0] = 1.0
		one_input = np.zeros(j)
		one_input[0] = 1.0
		one_answer = ubtph(one_input, j)
		try:
			assert np.all(np.isclose(one_answer, one_target))
		except AssertionError:
			raise AssertionError("one_answer: "+str(one_answer))

def test_lattice_in_unit_ball():
	lattice = lattice_in_unit_ball(3, 2)
	try:
		assert len(lattice) == 27
	except AssertionError:
		raise AssertionError("lattice: "+str(lattice)+"\n\n\nlength of lattice: "+str(len(lattice)))
	possible_vals = (-0.5, 0.0, 0.5)
	try:
		for a, b, c in it.product(possible_vals, repeat=3):
			assert (a, b, c) in lattice
	except AssertionError:
		raise AssertionError("lattice: "+str(lattice))

def test_permutation_matrix():
	dim = 6
	P = permutation_matrix(dim)
	original_vec = np.zeros(dim)
	original_vec[0] = 1.0
	for j in range(6):
		target_vec = np.zeros(dim)
		target_vec[j] = 1.0
		answer_vec = np.linalg.matrix_power(P, j) @ original_vec
		try:
			assert np.all(np.isclose(target_vec, answer_vec))
		except AssertionError:
			raise AssertionError("target_vec: "+str(target_vec)+"\n\n\nanswer_vec: "+str(answer_vec))

def test_parametrize_chart():
	dim = 6
	P = permutation_matrix(6)
	input_vec = np.zeros(6)
	input_vec[-1] = 1.0
	para, ortho = parametrize_chart(input_vec, P)
	assert len(para) == dim - 1
	for vec in para:
		cond = False
		for j in range(dim-1):
			target_vec = np.zeros(dim)
			target_vec[j] = 1.0
			if np.all(np.isclose(target_vec, vec)):
				cond = True
		if not cond:
			raise AssertionError("vec: "+str(vec)+"\n\n\n")
	assert np.all(ortho == input_vec)
