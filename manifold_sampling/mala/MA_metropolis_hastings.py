
import sys
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import scipy.linalg
import mala.metropolis_hastings as mh

# MANIFOLD-ADJUSTED METROPOLIS-HASTINGS ----------------------------------------
# meta-algorithm for markov-chain monte-carlo
# returns a list of positions attained
def MAMH(M,H,update_rule,initial_point,max_steps,**kwargs):
	trajectory=list()

	if 'time_step' in kwargs:
		time_span = np.linspace(0,max_steps,kwargs['time_step'])
	else:
		time_span = range(max_steps)

	observation_0 = Snapshot(0,pos=initial_point,
		grad=H.gradient(initial_point))
	observation_0.log_likelihood_grad(H)
	trajectory.append(observation_0)
	x = initial_point
	# at each time step
	for t in time_span:
		# suggest a candidate next sample
		# (we require this update rule be symmetric: p(x|y)=p(y|x))
		x_cand = update_rule(x,trajectory)
		#print(x_cand)
		# calculate the probability with which we'll accept the candidate
		accept_rate = mh.Hastings_ratio(x,x_cand,H)
		#print(accept_rate)
		# flip a coin with this weight
		if mh.weighted_coin(accept_rate):
			#print('accepted')
			x = x_cand
		# if we reject the candidate, x stays in its possition
		observation_t = Snapshot(t,pos=x,grad=H.gradient(x))
		observation_t.log_likelihood_grad(H)
		trajectory.append(observation_t)

	return trajectory

# TRAJECTORY DATA STORAGE ------------------------------------------------------
# a datastructure for storing info about our timesteps
class Snapshot():
	def __init__(self,t,**kwargs):
		self.t = t
		if 'pos' in kwargs: 
			self.pos = kwargs['pos']
		else:
			self.pos = None
		if 'grad' in kwargs: 
			self.grad = kwargs['grad']
		else:
			self.grad = None
		# Other attributes
		self.ll_grad = None

	def set_position(self,p):
		assert self.pos == None
		self.pos = p
		return

	def set_grad(self,g):
		assert self.grad == None
		self.grad = g
		return

	def log_likelihood_grad(self,potential):
		self.ll_grad = np.divide(potential.gradient(self.pos),
			potential.eval(self.pos))
		return

# UPDATE RULES -----------------------------------------------------------------
# Use the observed Fisher information matrix as an approximation of the 
# Fisher-Rao metric
def const_K_langevin(x,trajectory,metric_method,H,step_size,**kwargs):
	G = metric_method(x,trajectory)
	G_inv = np.linalg.inv(G)
	G_sqrt = np.linalg.cholesky(G)

	natural_grad = np.matmul(G_inv,H.gradient(x))

	random_momentum = np.matmul(G_sqrt,np.random.normal(loc=0.0,
		scale=1.0,size=x.size))

	return x + 0.5*np.power(step_size,2)*natural_grad+step_size*random_momentum

def empirical_Fisher_metric(H,x,traj):
	# if we have fewer observations than dimensions, we can't compute a 
	# nonsingular covariance matrix
	if len(traj)<=x.size:
		return np.identity(x.size)
	# a (no. observations) x (dimensions) matrix
	# each "observation" is the gradient of the log-likelihood at previous pts
	data = np.asarray([v.ll_grad for v in traj])

	# set rowvar=False because each row of data is an observation
	F = np.cov(data,rowvar=False)
	assert linalg.cond(F) < 1/sys.float_info.epsilon

	# "normalize"
	#F = np.divide(F,np.linalg.norm(F,ord='fro'))

	return F

# This would become G_hat = I + diag(x)K^T K diag(x)
def metric_tensor(M,x):
	E = np.identity(x.size)
	G = np.full(E.shape,np.nan)
	for ii in range(0,x.size):
		for jj in range(0,x.size):
			G[ii,jj] = M.inner(x,E[:,ii],E[:,jj])
	return G

def sphere_metric(M,H,x,step_size):
	G = metric_tensor(M,x)
	G_inv = np.linalg.inv(G)

	natural_grad = np.matmul(G_inv,H.gradient(x))

	return

	





