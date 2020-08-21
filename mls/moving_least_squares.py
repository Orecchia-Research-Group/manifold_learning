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

#Defining the weight function scaled by delta
def weight_scaled(x,delta):
    return weight(x/delta)

#Defining the weight function's derivative
def dweight(x):
    if (x <= -1) or (x >= 1):
        return 0
    elif (x < -1/2):
        return 4*(x+1)
    elif (x < 1/2):
        return (-4*x)
    elif (x < 1):
        return 4*(x-1)

#Defining the weight function's derivative scaled by delta
def dweight_scaled(x,delta):
    return dweight(x/delta)

#Defining the weight function's second derivative
def ddweight(x):
    if (x <= -1) or (x >= 1):
    	return 0
    elif (x < -1/2):
        return 4
    elif (x < 1/2):
        return -4
    elif (x < 1):
        return 4

#Defining the weight function's second derivative scaled by delta
def ddweight_scaled(x,delta):
    return ddweight(x/delta)

class C_1_MLS_oracle:
	"""
	The purpose of this class is to take a set of points (x_i,y_i)
	for x_i, y_i in R and to represent the optimal MLS approximant
	s_{f,X} of degree m for these points. It is implicitly assumed
	that there exists some f such that f(x_i) = y_i for all i
	"""
	def __init__(self, points, delta, m=2):
		"""
		points: set of n (x,y) pairs represented as a n x 2
			NumPy array
		m: the degree for which we require unisolvency (see
			chapter 4 of Wendland)
		delta: float, represents scaling factor
		"""
		# Defining instance variables delta, m, and points
		self.m = m
		self.delta = delta
		self.points = points

		# Default value for parameter d
		self.d = 0.01

		# Calculation of matrix P_complete
		total_rows = points.shape[1]
		P_complete = np.matrix(np.empty(shape=(total_rows,m+1)))		
		for i in range(0,m+1):
			for j in range(0, points.shape[1]):
				P_complete[j,i] = points[0,j]**i

		# Calculation of matrix F_complete
		F_complete = points[1]

		#Defining instance variables to be accessed later
		self.F = F_complete
		self.P = P_complete

	def eval(self, x):
		"""
		Given a real number x, return the MLS approximant for f
		at x, and first derivative x'.
		"""
		#Importing instance variables to use in calculations
		m = self.m
		delta = self.delta
		points = self.points
		#d = self.d 
		F_complete = self.F
		P_complete = self.P

	
		# Calculation of I_indices, array which includes indices j_0,...,j_n 
		# Calculation of I_values which includes x_(j_0),...,x_(j_n)
		list = []
		I_indices = np.array(list)
		I_values = np.array(list)

		total_rows = points.shape[1]
		for i in range(0, total_rows):
			if abs(points[0,i] - x) < delta:
				I_indices = np.append(I_indices,i)
				I_values = np.append(I_values,points[0,i])
		
		#Calculation of #(x), the size of the set I(x)
		pound = I_values.shape[0]
		if pound==0:
			return "Initial input x is too far from data for this value of delta. No values to compute."

		# Calculation of matrix R(x)
		R = np.empty(shape=(m + 1,1))
		for i in range(0,m + 1):
			if x == 0 and i == 0:
				R[i,0]=1
			else:
				R[i,0]=x**i
		
		## Calculation of matrix D, diagonal matrix with entries equal
		# to weight function evaluated at points in I_values 
		D = np.matrix(np.zeros(shape=(pound,pound)))
		for i in range(0, pound):
			D[i,i] = weight_scaled(I_values[i]-x,delta)
		is_all_zero = np.all((D == 0))
		if is_all_zero:
			return "Weight function returned all zeroes. Delta is too small."

		# Slicing matrix P_complete
		I_min = np.amin(I_indices)
		I_max = np.amax(I_indices)

		P = P_complete[int(I_min):int(I_max+1),]

		# Slicing matrix F_complete
		F = F_complete[int(I_min):int(I_max)+1]

		# Calculating the product p*(x)
		P_t = np.transpose(P)
		if P.shape[0] == 1:
			return "Initial input x is too far from data for this value of delta. Only one value to compute."
		try:
			inv = np.linalg.inv(np.matmul(np.matmul(P_t,D),P))
		except numpy.linalg.LinAlgError:
			return "Matrix is not invertible. Change input parameters."
		

		
		prod_1 = np.matmul(np.matmul(np.matmul(F,D), P), inv)
		p_star = np.matmul(prod_1,R)

		# Calculating derivative of matrix D
		D_prime = np.matrix(np.zeros(shape=(pound,pound)))
		for i in range(0, pound):
			D_prime[i,i] = dweight_scaled(I_values[i]-x,delta)

		# Calculating derivative of matrix R(x)
		R_prime = np.empty(shape=(m + 1,1))
		for i in range(0,m + 1):
			if x == 0 and i == 0:
				R_prime[i,0] = 0
			elif x == 0 and i == 1:
				R_prime[i,0] = 1
			else:
				R_prime[i,0]=i*(x**(i-1))

		# Calculating derivative of p*(x)
		first_term = np.matmul(np.matmul(np.matmul(D_prime, P), inv), R)
		second_term = np.subtract(R_prime, np.matmul(np.matmul(np.matmul( np.matmul(P_t,D_prime),P), inv),  R))
		third_term = np.matmul(np.matmul(D, P), inv)
		p_star_prime = np.matmul(F, np.add(first_term, np.matmul(third_term, second_term)))

		# Returns an array containing p*(x) as the first entry and the first derivative of p*(x) as the second entry
		return np.array([p_star[0,0], p_star_prime[0,0]])

	def insert(self, new_points):
		"""
		Given array of (x,y) tuples new_points, add these to the MLS calculation by changing 
		values of matrix P_complete and F_complete
		"""
		m = self.m
		points = self.points
		# Calculating new size
		new_size = points[0].shape[0] + new_points[0].shape[0]

		# Initializing new arrays
		new_x= np.zeros(new_size)
		new_y = np.zeros(new_size)
		new_arr = np.array([new_x,new_y])

		# Appending new values to current points
		new_arr[0] = np.append(points[0],new_points[0])
		new_arr[1] = np.append(points[1],new_points[1])

		#Changing value of instance variable
		self.points = new_arr

		# Calculation of matrix P_complete
		total_rows = new_arr.shape[1]
		P_complete = np.matrix(np.empty(shape=(total_rows,m+1)))		
		for i in range(0,m+1):
			for j in range(0, new_arr.shape[1]):
				P_complete[j,i] = new_arr[0,j]**i

		# Calculation of matrix F_complete
		F_complete = new_arr[1]

		#Defining instance variables to be accessed later
		self.F = F_complete
		self.P = P_complete
		pass




def main():
	pass

if __name__=="__main__":
	main()
