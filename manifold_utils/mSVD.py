import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds, LinearOperator
from scipy.sparse import vstack
import scipy
import fbpca
from tqdm import tqdm
from manifold_utils.power_iteration import iterated_power_method

def hypersphere(npoints, ndim):
    """
    This function creates samples on a unit sphere in desired dimensions.
    It requires the numpy package.
    Parameters:
        npoints (int): the number of points to be sampled
        ndim (int): the number of dimensions in which the unit sphere will be sampled
    """

    vec = np.random.randn(ndim,
                          npoints)  # creates a random sample from a Gaussian distribution in the form of an array of dimensions: ndim x npoints
    vec /= np.linalg.norm(vec,
                          axis=0)  # divides each vector by its norm, which turns each vector into a unit vector (length 1). Here we obtain samples from the unit sphere in the dimension we stated in the beginning of the function.
    return (vec)

class two_index_iterator:
    def __init__(self, thresholds, candidates, key=None):
        self.thresholds = thresholds
        self.candidates = candidates
        # Create auxiliary key function
        self.true_key = lambda x: x if not key else key(x)
        # initialize candidate indices

    def __iter__(self):
        cand_ind = 0
        for thresh in self.thresholds:
            new_candidates = []
            try:
                while self.true_key(self.candidates[cand_ind]) <= thresh:
                    new_candidates.append(self.candidates[cand_ind])
                    cand_ind += 1
                    if cand_ind == len(self.candidates):
                        break
            except IndexError:
                pass
            yield thresh, new_candidates

    def __len__(self):
        return len(self.thresholds)

### Function for obtaining eigenvalues while iterating through radii

def eigen_calc(cloud, center_ind, k, radint = .01):
    """
    This function iterates through specidic radii values and performs PCA at the given radius. The PCA values (eigenvalues, eigenvectors) are then saved and returned in a multidimensional list.
    Also, this function requires the numpy, random, and scipy packages for proper use.
    Parameters:
        cloud (arr): a multidimensional point cloud array that contains the coordinates of the points in the cloud
	dist_mat (arr): The entry at indices i,j is the Euclidean distance between points i and j
        center_ind (int): the index of the desired point on which the sphere is centered
        k (int): the intrinsic dimension of the data
        radint (float): Default = .01; the interval (step size) at which the radius expands
    """

    dim_array = np.shape(cloud)  # saves the dimensions of the array
    eigval_list = []  # creates empty list to store eigenvalues
    top_eigvecs = []  # creates empyt list in order to store the egenvectors of the intrinsic dimension
    dist_mat = np.zeros((dim_array[0],dim_array[0])) # creates an empty n x n matrix 
    
    # Fill in empty matrix with distances from each point to each other
    for i in range(dim_array[0]):
        for j in range(dim_array[0]):
            dist_mat[i,j] = np.linalg.norm(cloud[i,:]-cloud[j,:]) 

    # Select the distance vector to work with based on the center_ind
    dist_vec = dist_mat[center_ind,:].copy()
    sorted_vec = np.sort(dist_vec)
    indices = [*range(len(sorted_vec))]
    indices.sort(key=lambda x: dist_vec[x]) # sorts indices (of original points) in order from smallest distance to largest
    radii = [*np.arange(sorted_vec[5], sorted_vec[-1] + radint, radint)]
    shapes = [] # creates empty list to store shapes of X
    X = []

    for i in radii:
        for j in range(len(sorted_vec)):
            if (sorted_vec[j] <= i) and ((sorted_vec[j] > radii[radii.index(i)-1]) or (radii.index(i) == 0)) :
                X.append(cloud[indices[j], :])
        X_mat = np.vstack(X)  # creates a 'matrix' by vertically stacking the elements of the list
        dim_X = np.shape(X_mat)  # saves dimensions of matrix for points within the current radius
        shapes.append(dim_X)

        # Create the covariance matrix and save eigenvalues for each set X
        if radii.index(i) == 0:
            cov_X = np.cov(X_mat, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov_X)  # computes the eigenvalues and eigenvectors of the covariance matrix
            eigval_list.append(eigvals)  # appends the set of eigenvalues to the list created above
            top_eigvecs.append(eigvecs[0:k])
        elif shapes[radii.index(i)] != shapes[radii.index(i)-1]:
            cov_X = np.cov(X_mat, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov_X)  # computes the eigenvalues and eigenvectors of the covariance matr$
            eigval_list.append(eigvals)  # appends the set of eigenvalues to the list created above
            top_eigvecs.append(eigvecs[0:k])
        else:
            eigval_list.append(eigval_list[-1])
            top_eigvecs.append(top_eigvecs[-1])

    if len(eigval_list) == 0:
        raise ValueError(str(eigval_cache) + '\n\n\n' + str(top_eigvecs) + "\n\n\n" + str(radii))
    return[np.array(eigval_list),np.array(top_eigvecs),radii]


### Function for plotting eigenvalues obtained in the above function

def eigen_plot(eigval_list,radii, R_min, R_max):
    """
    This function plots the multidimensional eigenvalue list created from the eigen_calc function. X-axis corresponds to the radii value, while the y-axis corresponds to the eigenvalues. Each individual line represents a dimension.
    Also, this function requires both the matplotlib and numpy packages.
    Run the code: %matplotlib inline, when in jupyter notebook to display the plot in the notebook.
    Parameters:
        eigval_list (list): This is a multidimensional list containing eigenvalues at different radii values
        radint (int): the interval (step size) at which the radius expands
    """

    # Plot the eigenvalues
    #radii = np.arange(radstart, radend + radint, radint)  # creates an array of radii values to iterate through
    eig_mat = np.stack(eigval_list, axis=0)  # stacks eigenvalue list into an array (dimensions of N x D)
    dim_eig_mat = np.shape(eig_mat)  # saves dimensions of the eienvalue matrix for easy access
    fig = plt.figure()  # creates a figure plot
    axes = fig.add_subplot(111)  # adds x and y axes to the plot
    for i in range(dim_eig_mat[1]):  # iterates through the columns (dimensions) of the eigenvalue matrix
        axes.plot(radii, eig_mat[:, i])  # plots eigenvalues (y-axis) against each radii value (x-axis)
    #axes.axvspan(R_min, R_max, alpha=0.5, color = 'red')
    #st = "Manifold_radii" + str(np.random.randint(low=1, high = 1000)) + ".png"
    #plt.savefig(st)
    return (plt.show())


## Function for projecting epsilon-ball vectors onto a hyperplane

def eps_projection(vectors,eigvecs,center):
    """
    This function projects vectors within an epsilon ball onto a hyperplane defined be given eigenvectors.
    Parameters:
        vectors (arr): The set of points (vectors) to be projected onto the hyperplane
        eigvecs (arr): The set of eigenvectors which create the hyperplane
        center (arr): The center of both the epsilon ball and the hyperplane
    """
    
    newvecs = vectors-center
    projection=np.dot(newvecs,eigvecs)
    return(projection)

def eigen_calc_from_dist_mat(cloud, dist_mat, center_ind, radint = .01):
    """
    This function iterates through specidic radii values and performs PCA at the given radius. The PCA values (eigenvalues, eigenvectors) are then saved and returned in a multidimensional list.
    Also, this function requires the numpy, random, and scipy packages for proper use.
    Parameters:
        cloud (arr): a multidimensional point cloud array that contains the coordinates of the points in the cloud
	dist_mat (arr): The entry at indices i,j is the Euclidean distance between points i and j
        center_ind (int): the index of the desired point on which the sphere is centered
        radint (float): Default = .01; the interval (step size) at which the radius expands
    """
    N, d = cloud.shape # Get number N of points and dimension d of ambient space
    assert dist_mat.shape == (N, N) # Assert agreement between cloud.shape and dist_mat.shape

    dist_vec = dist_mat[center_ind, :]
    sorted_vec = np.sort(dist_vec)
    radii = [*np.arange(sorted_vec[5], sorted_vec[-1] + radint, radint)]
    indices = list(range(N))
    indices.sort(key=lambda x: dist_vec[x])

    eigval_list = []
    eigvec_list = []
    for rad, cands in two_index_iterator(radii, indices, key=lambda x: dist_vec[x]):
        if len(cands) > 0:
            new_cands = np.stack([cloud[cand, :] for cand in cands], axis=0)
            try:
                points = np.vstack([points, new_cands])
            except NameError:
                points = new_cands
            cov_X = np.cov(points, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov_X)
            eigval_list.append(eigvals)
            eigvec_list.append(eigvecs)
        else:
            eigval_list.append(eigval_list[-1])
            eigvec_list.append(eigvec_list[-1])

    return radii, eigval_list, eigvec_list

def eigen_plot_numPoints(eigval_list, xaxis, xtype, cid):
    """
    This function plots the multidimensional eigenvalue list created from the eigen_calc function. X-axis corresponds to the radii value, while the y-axis corresponds to the eigenvalues. Each individual line represents a dimension.
    Also, this function requires both the matplotlib and numpy packages.
    Run the code: %matplotlib inline, when in jupyter notebook to display the plot in the notebook.

    Parameters:
        eigval_list (list): This is a multidimensional list containing eigenvalues at different radii values
        x_axis. Either radii or number of points included
    """

    # Plot the eigenvalues
    #radii = np.arange(radstart, radend + radint, radint)  # creates an array of radii values to iterate through
    eig_mat = np.stack(eigval_list, axis=0)  # stacks eigenvalue list into an array (dimensions of N x D)
    dim_eig_mat = np.shape(eig_mat)  # saves dimensions of the eienvalue matrix for easy access
    fig = plt.figure()  # creates a figure plot
    axes = fig.add_subplot(111)  # adds x and y axes to the plot
    for i in range(dim_eig_mat[1]):  # iterates through the columns (dimensions) of the eigenvalue matrix
        axes.plot(xaxis, eig_mat[:, i])  # plots eigenvalues (y-axis) against each radii value (x-axis)
    #axes.axvspan(R_min, R_max, alpha=0.5, color = 'red')
    st = xtype + 'cid=' + str(cid) + ".jpg"
    plt.xlabel(xtype, fontsize=14)
    plt.ylabel('Eigenvalues')
    plt.savefig(st, dpi=400)
    return (plt.show())

def eigen_calc_from_dist_mat_withNumPoints(cloud, dist_mat, center_ind, Rstart, Rend, radint = .01):
    """
    This function iterates through specidic radii values and performs PCA at the given radius. The PCA values (eigenvalues, eigenvectors) are then saved and returned in a multidimensional list.
    Also, this function requires the numpy, random, and scipy packages for proper use.

    Parameters:
        cloud (arr): a multidimensional point cloud array that contains the coordinates of the points in the cloud
	dist_mat (arr): The entry at indices i,j is the Euclidean distance between points i and j
        center_ind (int): the index of the desired point on which the sphere is centered
        radint (float): Default = .01; the interval (step size) at which the radius expands
    """
    N, d = cloud.shape # Get number N of points and dimension d of ambient space
    assert dist_mat.shape == (N, N) # Assert agreement between cloud.shape and dist_mat.shape

    dist_vec = dist_mat[center_ind, :]
    sorted_vec = np.sort(dist_vec)
    radii = [*np.arange(sorted_vec[5], sorted_vec[-1] + radint, radint)]
    indices = list(range(N))
    indices.sort(key=lambda x: dist_vec[x])
    
    radius_list = []
    eigval_list = []
    eigvec_list = []
    numPoints_list = [] #track the number of points 
    for rad, cands in two_index_iterator(radii, indices, key=lambda x: dist_vec[x]):  
        if len(cands) > 0:
            new_cands = np.stack([cloud[cand, :] for cand in cands], axis=0)
            try:
                points = np.vstack([points, new_cands])
            except NameError:
                points = new_cands
            if rad < Rstart:
                continue;
            elif rad > Rend:
                break;
            else:
                cov_X = np.cov(points, rowvar=False)
                eigvals, eigvecs = np.linalg.eigh(cov_X)
                eigval_list.append(eigvals)
                eigvec_list.append(eigvecs)
                numPoints_list.append(len(points))
                radius_list.append(rad)
        else:
            if rad < Rstart:
                continue;
            elif rad > Rend:
                break;
            else:
                eigval_list.append(eigval_list[-1])
                eigvec_list.append(eigvec_list[-1])
                numPoints_list.append(numPoints_list[-1])
                radius_list.append(rad)
    return radius_list, numPoints_list, eigval_list, eigvec_list

def rapid_eigen_calc_from_dist_mat(cloud, dist_mat, center_ind, radint = .01, k=10, norm_diff=1e-6):
    """
    This function iterates through specidic radii values and performs PCA at the given radius. The PCA values (eigenvalues, eigenvectors) are then saved and returned in a multidimensional list.
    Also, this function requires the numpy and random packages for proper use.
    Parameters:
        cloud (arr): a multidimensional point cloud array that contains the coordinates of the points in the cloud
	dist_mat (arr): The entry at indices i,j is the Euclidean distance between points i and j
        center_ind (int): the index of the desired point on which the sphere is centered
        radint (float): Default = .01; the interval (step size) at which the radius expands
    """
    N, d = cloud.shape # Get number N of points and dimension d of ambient space
    assert dist_mat.shape == (N, N) # Assert agreement between cloud.shape and dist_mat.shape

    dist_vec = dist_mat[center_ind, :]
    sorted_vec = np.sort(dist_vec)
    radii = [*np.arange(sorted_vec[5], sorted_vec[-1] + radint, radint)]
    indices = list(range(N))
    indices.sort(key=lambda x: dist_vec[x])

    eigval_list = []
    eigvec_list = []
    for rad, cands in two_index_iterator(radii, indices, key=lambda x: dist_vec[x]):
        if len(cands) > 0:
            new_cands = np.stack([cloud[cand, :] for cand in cands], axis=0)
            try:
                points = np.vstack([points, new_cands])
            except NameError:
                points = new_cands
            cov_X = np.cov(points, rowvar=False)
            eigvals, eigvecs = iterated_power_method(cov_X, k=k, norm_diff=1e-6)
            eigval_list.append(eigvals)
            eigvec_list.append(eigvecs)
        else:
            eigval_list.append(eigval_list[-1])
            eigvec_list.append(eigvec_list[-1])

    return radii, eigval_list, eigvec_list

def Sparse_eigen_calc_from_dist_mat(cloud, dist_mat, center_ind, Rend, radint=.01, k=10):
    """
    This function iterates through specidic radii values and performs PCA at the given radius. The PCA values (eigenvalues, eigenvectors) are then saved and returned in a multidimensional list.
    Also, this function requires the numpy, random, and scipy packages for proper use.

    Parameters:
        cloud (arr): a multidimensional point cloud array that contains the coordinates of the points in the cloud
        center_ind (int): the index of the desired point on which the sphere is centered
        radstart (int): the first radius value of the expanding sphere
        radend (int): the final value (included) of the expanding spherical radius
        radint (float): Default = .01; the interval (step size) at which the radius expands
    """
    N, d = cloud.shape # Get number N of points and dimension d of ambient space
    assert dist_mat.shape == (N, N) # Assert agreement between cloud.shape and dist_mat.shape

    dist_vec = dist_mat[center_ind, :]
    sorted_vec = np.sort(dist_vec)
    radii = np.arange(sorted_vec[k], Rend + radint, radint)
    indices = list(range(N))
    indices.sort(key=lambda x: dist_vec[x])
    radius_list = []
    eigval_list = []
    eigvec_list = []
    numPoints_list = [] #track the number of points 
    for rad, cands in two_index_iterator(radii, indices, key=lambda x: dist_vec[x]):
        if len(cands) > 0:
            new_cands = vstack([cloud[cand, :] for cand in cands])
            try:
                points = vstack([points, new_cands])
            except NameError:
                points = new_cands
            if rad > Rend:
                break;
            else:
                # Get sample size
                n = points.shape[0]
                #svd for the top k
                u, s, vt = svds(points, k)
                eigvals = np.square(s)/(n-1)
                radius_list.append(rad)
                eigval_list.append(np.square(s)/(n-1))
                eigvec_list.append(vt)
                numPoints_list.append(n)
        else:
            if rad > Rend:
                break;
            else:
                eigval_list.append(eigval_list[-1])
                eigvec_list.append(eigvec_list[-1])
                numPoints_list.append(numPoints_list[-1])
                radius_list.append(rad)
    return radius_list, numPoints_list, eigval_list, eigvec_list

def get_centered_sparse(points, xbar, dtype=np.float64):
	N, d = points.shape
	points_T = points.transpose()
	def matvec(v):
		return points.dot(v) - xbar.dot(v)*np.ones(N)
	def rmatvec(v):
		return points_T.dot(v) - np.sum(v)*xbar
	def matmat(V):
		return points.dot(V) - np.vstack([xbar.dot(V)]*N)
	def rmatmat(V):
		return points_T.dot(V) - np.outer(xbar, np.sum(V, axis=0))
	return LinearOperator(points.shape, matvec, rmatvec, matmat, dtype, rmatmat)

def Sparse_eigen_calc_from_dist_mat_uncentered(cloud, dist_mat, center_ind, Rend, radint=.01, k=10):
    """
    This function iterates through specidic radii values and performs PCA at the given radius. The PCA values (eigenvalues, eigenvectors) are then saved and returned in a multidimensional list.
    Also, this function requires the numpy, random, and scipy packages for proper use.

    Parameters:
        cloud (arr): a multidimensional point cloud array that contains the coordinates of the points in the cloud
        center_ind (int): the index of the desired point on which the sphere is centered
        radstart (int): the first radius value of the expanding sphere
        radend (int): the final value (included) of the expanding spherical radius
        radint (float): Default = .01; the interval (step size) at which the radius expands
    """
    N, d = cloud.shape # Get number N of points and dimension d of ambient space
    assert dist_mat.shape == (N, N) # Assert agreement between cloud.shape and dist_mat.shape

    dist_vec = dist_mat[center_ind, :]
    sorted_vec = np.sort(dist_vec)
    radii = np.arange(sorted_vec[k], Rend + radint, radint)
    indices = list(range(N))
    indices.sort(key=lambda x: dist_vec[x])
    radius_list = []
    eigval_list = []
    eigvec_list = []
    numPoints_list = [] #track the number of points 
    for rad, cands in tqdm(two_index_iterator(radii, indices, key=lambda x: dist_vec[x])):
        if len(cands) > 0:
            new_cands = vstack([cloud[cand, :] for cand in cands])
            try:
                points = vstack([points, new_cands])
            except NameError:
                points = new_cands
            if rad > Rend:
                break;
            else:
                # Get sample size
                n = points.shape[0]
                # Get sample mean
                xbar = points.mean(axis=0)
                # Get centered "sparse" matrix
                A = get_centered_sparse(points, xbar)
                #svd for the top k
                u, s, vt = svds(A, k)
                eigvals = np.square(s)/(n-1)
                radius_list.append(rad)
                eigval_list.append(np.square(s)/(n-1))
                eigvec_list.append(vt)
                numPoints_list.append(n)
        else:
            if rad > Rend:
                break;
            else:
                eigval_list.append(eigval_list[-1])
                eigvec_list.append(eigvec_list[-1])
                numPoints_list.append(numPoints_list[-1])
                radius_list.append(rad)
    return radius_list, numPoints_list, eigval_list, eigvec_list

def eigen_calc_from_dist_vec(cloud, dist_vec, Rend, radint=.01, k=10, n_iter=2):
    """
    This function iterates through specidic radii values and performs PCA at the given radius. The PCA values (eigenvalues, eigenvectors) are then saved and returned in a multidimensional list.
    Also, this function requires the numpy, random, and scipy packages for proper use.

    Parameters:
        cloud (arr): a multidimensional point cloud array that contains the coordinates of the points in the cloud
        center_ind (int): the index of the desired point on which the sphere is centered
        radstart (int): the first radius value of the expanding sphere
        radend (int): the final value (included) of the expanding spherical radius
        radint (float): Default = .01; the interval (step size) at which the radius expands
    """
    N, d = cloud.shape # Get number N of points and dimension d of ambient space
    assert dist_vec.shape == (N,) # Assert agreement between cloud.shape and dist_mat.shape

    sorted_vec = np.sort(dist_vec)
    radii = np.arange(sorted_vec[k], Rend + radint, radint)
    indices = list(range(N))
    indices.sort(key=lambda x: dist_vec[x])
    radius_list = []
    eigval_list = []
    eigvec_list = []
    numPoints_list = [] #track the number of points 
    for rad, cands in tqdm(two_index_iterator(radii, indices, key=lambda x: dist_vec[x])):
        if len(cands) > 0:
            new_cands = np.vstack([cloud[cand, :] for cand in cands])
            try:
                points = vstack([points, new_cands])
            except NameError:
                points = new_cands
            if rad > Rend:
                break;
            else:
                # Get sample size
                n = points.shape[0]
                #svd for the top k
                u, s, vt = fbpca.pca(points, k=5, n_iter=n_iter)
                vt = vt.T
                eigvals = np.square(s)/(n-1)
                radius_list.append(rad)
                eigval_list.append(np.square(s)/(n-1))
                eigvec_list.append(vt)
                numPoints_list.append(n)
        else:
            if rad > Rend:
                break;
            else:
                eigval_list.append(eigval_list[-1])
                eigvec_list.append(eigvec_list[-1])
                numPoints_list.append(numPoints_list[-1])
                radius_list.append(rad)
    return radius_list, numPoints_list, eigval_list, eigvec_list
