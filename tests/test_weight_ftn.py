import numpy as np
from mls.moving_least_squares import weight, weight_scaled, dweight, dweight_scaled, ddweight, ddweight_scaled

def test_weight():
	"""
	Tests that the weight function returns correct values 
	in range (-1,1) and 0 otherwise
	"""
	input_vals = np.array([-2,-.75,0.25,.6,2])
	output = np.empty(len(input_vals))
	expected_vals = np.array([0,0.125,0.875,0.32,0])
	
	for i in range(0, len(input_vals)):
		output[i] = weight(input_vals[i])
		assert np.isclose(output[i], expected_vals[i])

def test_dweight():
	"""
	Tests that the dweight function returns correct values 
	in range (-1,1) and 0 otherwise
	"""
	input_vals = np.array([-2,-.75,0.25,.6,2])
	output = np.empty(len(input_vals))
	expected_vals = np.array([0,1,-1,-1.6,0])

	for i in range(0, len(input_vals)):
		output[i] = dweight(input_vals[i])
		assert np.isclose(output[i], expected_vals[i])


def test_ddweight():
	"""
	Tests that the ddweight function returns correct values 
	in range (-1,1) and 0 otherwise
	"""
	input_vals = np.array([-2,-.75,0.25,.6,2])
	output = np.empty(len(input_vals))
	expected_vals = np.array([0,4,-4,4,0])

	for i in range(0, len(input_vals)):
		output[i] = ddweight(input_vals[i])
		assert np.isclose(output[i], expected_vals[i])


def test_weight_scaled():
	"""
	Tests that the weight_scaled function returns correct values 
	in range (-1,1) and 0 otherwise
	"""
	input_vals = np.array([-4,-1.5,0.5,1.2,4])
	output = np.empty(len(input_vals))
	expected_vals = np.array([0,0.125,0.875,0.32,0])

	for i in range(0, len(input_vals)):
		output[i] = weight_scaled(input_vals[i],2)
		assert np.isclose(output[i], expected_vals[i])

def test_dweight_scaled():
	"""
	Tests that the dweight_scaled function returns correct values 
	in range (-1,1) and 0 otherwise
	"""
	input_vals = np.array([-4,-1.5,0.5,1.2,4])
	output = np.empty(len(input_vals))
	expected_vals = np.array([0,1,-1,-1.6,0])

	for i in range(0, len(input_vals)):
		output[i] = dweight_scaled(input_vals[i],2)
		assert np.isclose(output[i], expected_vals[i])

def test_ddweight_scaled():
	"""
	Tests that the ddweight_scaled function returns correct values 
	in range (-1,1) and 0 otherwise
	"""
	input_vals = np.array([-4,-1.5,0.5,1.2,4])
	output = np.empty(len(input_vals))
	expected_vals = np.array([0,4,-4,4,0])

	for i in range(0, len(input_vals)):
		output[i] = ddweight_scaled(input_vals[i],2)
		assert np.isclose(output[i], expected_vals[i])
