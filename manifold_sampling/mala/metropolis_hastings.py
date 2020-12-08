
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import scipy.linalg

# METROPOLIS-HASTINGS ----------------------------------------------------------
# meta-algorithm for markov-chain monte-carlo
# returns a list of positions attained
def metropolis(potential,update_rule,initial_point,max_steps,**kwargs):
	trajectory=list()

	if 'time_step' in kwargs:
		time_span = np.linspace(0,max_steps,kwargs['time_step'])
	else:
		time_span = range(max_steps)

	trajectory.append(initial_point)
	x = initial_point
	# at each time step
	for _ in time_span:
		# suggest a candidate next sample
		# (we require this update rule be symmetric: p(x|y)=p(y|x))
		x_cand = update_rule(potential,x,**kwargs)
		#print(x_cand)
		# calculate the probability with which we'll accept the candidate
		accept_rate = Hastings_ratio(x,x_cand,potential)
		#print(accept_rate)
		# flip a coin with this weight
		if weighted_coin(accept_rate):
			#print('accepted')
			x = x_cand
		# if we reject the candidate, x stays in its possition
		trajectory.append(x)

	return trajectory

# calculate the probability with which we'll accept the candidate
def Hastings_ratio(x,x_prime,potential):
	return min([1.0,potential.eval(x_prime)/potential.eval(x)])

# flip a weighted coin
def weighted_coin(w):
	x = np.random.uniform(low=0.0,high=1.0)
	return w>x

# UPDATE RULES -----------------------------------------------------------------
# It is here, in our choice of a candidate next step, that we determine the 
# dynamics of our markov chain. Our update rule determines some transition 
# probability, p(y|x) the probability of moving to y from position x. Since we 
# are using the Hastings ratio to determine our acceptance rate, we must make 
# sure that our proposal/update rule is symmetric: the conditional probability 
# of moving from x_1 to x_2,  p(x_1 | x_2), must be equal to the conditional 
# probability of moving from x_2 to x_1, p(x_2 | x_1).  The most vanilla version
# simply takes a Gaussian of some radius arround our initial point

# Most of our core update rules are independent of time. However, there are
# methods that changing step size or other dynamics with time to achieve better
# convergence

# generate a candidate step by sampling an entry-wise gaussian around x
def gaussian(_,x,**kwargs):
	radius = kwargs['radius']
	return np.array([np.random.normal(loc=x[ii],scale=radius) 
		for ii in range(x.size)])

# First-order Langevin update. Intiuitively: our Langevin update rule has us 
# drift in the direction of the gradient*, with a small "kick" by some random 
# isotropic force

# We're actually going to move in the direction of the gradient of log(f), if
# f is our potential function. This doesn't change the direction of our gradient
# contribution, it just normalizes it against the function value at that point.
# Note that by chain rule we don't need any new information to compute this
def langevin(H,x,step_size,**kwargs):
	random_momentum = np.random.normal(loc=0.0,scale=1.0,size=x.size)
	grad = np.divide(H.gradient(x),H.eval(x))
	return x + np.power(step_size,2)*grad+ step_size*random_momentum
	





