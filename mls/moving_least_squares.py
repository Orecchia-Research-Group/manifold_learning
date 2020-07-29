import numpy as np
import scipy 
import matplotlib.pyplot as plt

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

		#Generating weight function: cubic spline
		x = np.linspace(-1,1,100)
		y1 = -2*np.power(x,2) + 1
		y2=2*np.power(np.add(x,1),2)
		y3=2*np.power(np.subtract(x,1),2)
		y = np.concatenate((y2[0:25],y1[25:75],y3[75:100]))
		from scipy.interpolate import interp1d

		#w(x) returns the value of the weight function for a value x in [-1,1]
		w = interp1d(x, y, kind='cubic')






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
