import numpy as np

class C_1_MLS_oracle:
	"""
	The purpose of this class is to take a set of points (x_i,y_i)
	for x_i, y_i in R and to represent the optimal MLS approximant
	s_{f,X} of degree m for these points. It is implicitly assumed
	that there exists some f such that f(x_i) = y_i for all i.
	"""
	def __init__(self, points, m=10):
		"""
		points: set of n (x,y) pairs represented as a n x 2
			NumPy array
		m: the degree for which we require unisolvency (see
			chapter 4 of Wendland)
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
