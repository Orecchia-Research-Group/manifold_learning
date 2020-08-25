import numpy as np
from pymanopt.manifolds import Grassmann, PSDFixedRank, Rotations, Sphere, Stiefel, Oblique
import pickle as pkl
from tqdm import tqdm

def sample_grassmann(d, p, n):
	"""
	Sample n points from the manifold of p-dimensional subspaces
	of R^d using pymanopt.manifolds.grassmann.Grassmann.rand.

	The resulting point cloud will represent a manifold of (n-p)*p
	dimensions.
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
	rank k using pymanopt.manifolds.psd.PSDFixedRank.rand. The
	resulting point cloud will represent a manifold of dimension
	((2*d+1)*k-k**2) / 2.
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
	pymanopt.manifolds.Rotations.rand.

	The resulting point cloud will represent a manifold with d*(d-1)/2 dimensions.
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



def sample_sphere(n,d):
    """
    Samples n points in a d-dimensional unit sphere. The resulting point cloud
    will represent a (d-1)-dimensional manifold.
    """
    try:
        with open("data/sphere__"+str(n)+"_"+str(d)+"_"+".pkl", "rb") as f:
            points = pkl.load(f)
    except FileNotFoundError:
        print("Sampling "+str(n)+" points from Sphere("+str(d)+")...")
        points=[]
        manifold=Sphere(d)
 
        for i in tqdm(range(n)):
            points.append(manifold.rand())
        points=np.stack(points)
        with open("data/sphere__"+str(n)+"_"+str(d)+"_"+".pkl", "wb") as f:
            pkl.dump(points, f)
    return points



def sample_stiefel(numsamples,n,p,k=1):
    """
    Sample n points from the n x p Stiefel matrices using
    pymanopt.manifolds.Stiefel.rand. The resulting point cloud will represent a
    manifold of n*p - p*(p+1)/2 dimensions.
    """
    try:
        with open("data/stiefel__"+str(numsamples)+"_"+str(n)+"_"+str(p)+"_"+str(k)+".pkl", "rb") as f:
            points = pkl.load(f)
    except FileNotFoundError:
        points=[]
        manifold=Stiefel(n,p,k=1)

        for i in tqdm(range(numsamples)):
            points.append(np.reshape(manifold.rand(),(n*p)))
        points=np.stack(points)
        with open("data/stiefel__"+str(numsamples)+"_"+str(n)+"_"+str(p)+"_"+str(k)+".pkl", "wb") as f:
            pkl.dump(points, f)
    return points


def sample_oblique(numsamples,m,n):
    """
    Sample n points from the m x n oblique matrices using
    pymanopt.manifolds.Oblique.rand. The resulting point cloud will represent a
    manifold of (m-1)*n dimensions.
    """
    try:
        with open("data/oblique__"+str(numsamples)+"_"+str(m)+"_"+str(n)+".pkl", "rb") as f:
            points = pkl.load(f)
    except FileNotFoundError:
        points=[]
        manifold=Oblique(m,n)

        for i in tqdm(range(numsamples)):
            points.append(np.reshape(manifold.rand(),(m*n)))
        points=np.stack(points)
        with open("data/oblique__"+str(numsamples)+"_"+str(m)+"_"+str(n)+".pkl", "wb") as f:
            pkl.dump(points, f)
    return points
