import numpy as np
from pymanopt.manifolds import Grassmann, PSDFixedRank, Rotations
import pickle as pkl
from tqdm import tqdm

def sample_grassmann(d, p, n):
	"""
	Sample n points from the manifold of p-dimensional subspaces
	of R^d using pymanopt.manifolds.grassmann.Grassmann.rand
	"""
	try:
		with open("data/grassmann__"+str(d)+"_"+str(p)+"_"+str(n)+".pkl", "rb") as f:
			points = pkl.load(f)
	except FileNotFoundError:
		print("Sampling "+str(n)+" points from Grassmann(+"+str(d)+","+str(p)+")...")
		manifold = Grassmann(d, p)
		points = []
		for _ in tqdm(range(n)):
			points.append(manifold.rand())
		points = np.stack(points)
		with open("data/grassmann__"+str(d)+"_"+str(p)+"_"+str(n)+".pkl", "wb") as f:
			pkl.dump(points, f)
	return points

def sample_psd_fixed_rank(d, k, n, norm_bound=None):
	"""
	Sample n points from the manifold of d x d PSD matrices of
	rank k using pymanopt.manifolds.psd.PSDFixedRank.rand
	"""
	if not norm_bound:
		norm_bound = np.inf
	try:
		with open("data/psdfr__"+str(d)+"_"+str(k)+"_"+str(n)+"_"+str(norm_bound)+".pkl", "rb") as f:
			points = pkl.load(f)
	except FileNotFoundError:
		print("Sampling "+str(n)+" points from PSDFixedRank("+str(n)+","+str(k)+")...")
		print("Rejecting all points with Frobenius norm greater than "+str(norm_bound)+"...")
		manifold = PSDFixedRank(d, k)
		points = []
		for _ in tqdm(range(n)):
			cond = False
			while not cond:
				point = manifold.rand()
				if np.linalg.norm(point, "fro") < norm_bound:
					points.append(manifold.rand())
					cond = True
		points = np.stack(points)
		with open("data/psdfr__"+str(d)+"_"+str(k)+"_"+str(n)+"_"+str(norm_bound)+".pkl", "wb") as f:
			pkl.dump(points, f)
	return points

def sample_rotations_manifold(d, n):
	"""
	Sample n points from SO(d) using
	pymanopt.manifolds.Rotations.rand
	"""
	try:
		with open("data/rotations__"+str(d)+"_"+str(n)+".pkl", "rb") as f:
			points = pkl.load(f)
	except FileNotFoundError:
		print("Sampling "+str(n)+" points from Rotations("+str(d)+")...")
		manifold = Rotations(d)
		points = []
		for _ in tqdm(range(n)):
			points.append(manifold.rand())
		points = np.stack(points)
		with open("data/rotations__"+str(d)+"_"+str(n)+".pkl", "wb") as f:
			pkl.dump(points, f)
	return points
