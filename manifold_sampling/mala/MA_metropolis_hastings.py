
import sys
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import scipy.linalg
import mala.metropolis_hastings as mh
import mala.icosehedron as ico

# MANIFOLD-ADJUSTED METROPOLIS-HASTINGS ----------------------------------------
# meta-algorithm for markov-chain monte-carlo, moving in ambient space
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

# ICOSAHEDRON MANIFOLD-ADJUSTED METROPOLIS-HASTINGS ----------------------------
class ico_point:
	def __init__(self,chart_coords,face_node_idx,face_map):
		self.face_node = face_node_idx
		self.face_obj = face_map[face_node_idx]

		self.face_coors = chart_coords
		self.euclidean_coors = ico.chart2euclidean(chart_coords,self.face_obj)

def ico_MAMH(H,max_steps,step_size):
	face_graph,vetex_graph,face_dict = ico.generate_icosahedron()

	trajectory=list()

	if 'time_step' in kwargs:
		time_span = np.linspace(0,max_steps,kwargs['time_step'])
	else:
		time_span = range(max_steps)

	# chose some random initial point
	# right now this isn't random, it's just arbitrary. we're initializing in
	# the face_node corresponding to index 0, at the origin wrt that faces' 
	# chart
	x = ico_point(face_coords=np.array([0,0]),
		face_node_idx =0,
		face_map=face_dict)
	trajectory.append(x)

	# at each time step
	for t in time_span:
		# suggest a candidate next sample
		# (we require this update rule be symmetric: p(x|y)=p(y|x))
		x_cand_face_coors = icosehedron_langevin(x,H,step_size)
		x_cand = ico_point(face_coords=x_cand_face_coors,
			face_node_idx=x.face_node_idx,
			face_map=face_dict)

		# calculate the probability with which we'll accept the candidate
		accept_rate = mh.Hastings_ratio(x.euclidean_coors,x_cand.euclidean_coors,H)
		#print(accept_rate)
		# flip a coin with this weight
		if mh.weighted_coin(accept_rate):
			x = x_cand
		# check if we've left our face
		if not np.all(ico.check_if_point_in_face(x.face_coords,x.face_obj)):
			# find the next face we'll move to
			next_face_node = face_across_edge(x.face_obj,x.face_coords,face_graph)
			next_face_obj = face_dict[next_face_node]

			# get coordinates of x in the chart of our next face
			next_chart_coors = map_pt_btwn_charts(x.face_coords,x.face_obj,next_face_obj)

			# update x
			x = ico_point(face_coords=next_chart_coors,
				face_node_idx =next_face_node,
				face_map=face_dict)
		# if we reject the candidate, x stays in its possition
		trajectory.append(x)

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

	natural_grad = np.matmul(G_inv,np.divide(H.gradient(x),H.eval(x)))

	random_momentum = np.matmul(G_sqrt,np.random.normal(loc=0.0,
		scale=1.0,size=x.size))

	return x + 0.5*np.power(step_size,2)*natural_grad+step_size*random_momentum

def icosehedron_langevin(x,H,step_size):
	G = ico_metric(x.face_coors)
	G_inv = np.linalg.inv(G)
	G_sqrt = np.linalg.cholesky(G)

	# map our point to euclidean space, take the gradient, and project the
	# gradient back into our chart
	chart_gradient = ico.euclidean2chart(H.gradient(x.euclidean_coors),x.face_obj)
	natural_grad = np.matmul(G_inv,np.divide(chart_gradient,H.eval(x.euclidean_coors)))

	# take random impulse in R^2
	random_momentum = np.matmul(G_sqrt,np.random.normal(loc=0.0,
		scale=1.0,size=x.face_coors.size))

	return x.face_coors + 0.5*np.power(step_size,2)*natural_grad+step_size*random_momentum

# given an array in chart coordinates, return estinamted metric tensor
def ico_metric(x):
	sphere_radius = np.sqrt([1+np.power(phi,2)])
	K = np.array([1/sphere_radius,1/sphere_radius])

	return np.identity(2)+np.diag(x)@np.transpose(K)@K@np.diag(x)

def empirical_Fisher_metric(H,x,traj,burnin):
	# if we have fewer observations than dimensions, we can't compute a 
	# nonsingular covariance matrix
	# add a buffer to collect more steps because unlucky first iterations
	# can really mess us up
	if len(traj)<=burnin:
		return np.identity(x.size)
	# a (no. observations) x (dimensions) matrix
	# each "observation" is the gradient of the log-likelihood at previous pts
	data = np.asarray([v.ll_grad for v in traj])

	# set rowvar=False because each row of data is an observation
	F = np.cov(data,rowvar=0)
	# this is the check numpy will use to determine invertibility
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



