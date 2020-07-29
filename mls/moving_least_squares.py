import numpy as np
import scipy 
import matplotlib.pyplot as plt


#Defining the weight function
def weight(x):
    if (x <= -1) or (x >= 1):
        return 0
    elif (x < -1/2):
        return 2*((x+1)**2)
    elif (x < 1/2):
        return -2*(x**2) + 1
    elif (x < 1):
        return 2*((x-1)**2)

def weight_scaled(x,delta):
    return weight(x/delta)

def dweight(x):
    if (x <= -1) or (x >= 1):
        return 0
    elif (x < -1/2):
        return 4*(x+1)
    elif (x < 1/2):
        return (-4*x)
    elif (x < 1):
        return 4*(x-1)

def dweight_scaled(x,delta):
    return dweight(x/delta)

def ddweight(x):
    if (x <= -1) or (x >= 1):
        return 0
    elif (x < -1/2):
        return 4
    elif (x < 1/2):
        return -4
    elif (x < 1):
        return 4

def ddweight_scaled(x,delta):
    return ddweight(x/delta)

class C_1_MLS_oracle:
	"""
	The purpose of this class is to take a set of points (x_i,y_i)
	for x_i, y_i in R and to represent the optimal MLS approximant
	s_{f,X} of degree m for these points. It is implicitly assumed
	that there exists some f such that f(x_i) = y_i for all i
	"""
	def __init__(self, points, delta, m=10):
		"""
		points: set of n (x,y) pairs represented as a n x 2
			NumPy array
		m: the degree for which we require unisolvency (see
			chapter 4 of Wendland)
		delta: float, represents scaling factor
		"""



		## Stuff


		pass

	def eval(self, x):
		"""
		Given a real number x, return the MLS approximant for f
		at x.
		"""
		## Stuff
		pass

	def slope(self, x):
		"""
		Given a real number x, return the derivative for the MLS
		approximant for f at x.
		"""
		pass
