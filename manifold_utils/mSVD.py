import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.spatial import distance
import matplotlib.pyplot as plt


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


### Function for obtaining eigenvalues while iterating through radii

def eigen_calc(cloud, center_ind, k, radint = .01):
    """
    This function iterates through specidic radii values and performs PCA at the given radius. The PCA values (eigenvalues, eigenvectors) are then saved and returned in a multidimensional list.
    Also, this function requires the numpy, random, and scipy packages for proper use.

    Parameters:
        cloud (arr): a multidimensional point cloud array that contains the coordinates of the points in the cloud
        center_ind (int): the index of the desired point on which the sphere is centered
        k (int): the intrinsic dimension of the data
	    # radstart (int): the first radius value of the expanding sphere
        # radend (int): the final value (included) of the expanding spherical radius
        radint (int): Default = .01; the interval (step size) at which the radius expands
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
    indices.sort(key=lambda x: sorted_vec[x]) # sorts indices (of original points) in order from smallest distance to largest
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
            eigvals, eigvecs = np.linalg.eigh(cov_X)  # computes the eigenvalues and eigenvectors of the covariance matrix and stores them in t$
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

def eigen_plot(eigval_list,radii):
    """
    This function plots the multidimensional eigenvalue list created from the eigen_calc function. X-axis corresponds to the radii value, while the y-axis corresponds to the eigenvalues. Each individual line represents a dimension.
    Also, this function requires both the matplotlib and numpy packages.
    Run the code: %matplotlib inline, when in jupyter notebook to display the plot in the notebook.

    Parameters:
        eigval_list (list): This is a multidimensional list containing eigenvalues at different radii values
    	radstart (int): the first radius value of the expanding sphere
        radend (int): the final value (included) of the expanding spherical radius
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

    return (plt.show())


## Function for projecting epsilon-ball vectors onto a hyperplane

def eps_projection(vectors,eigvecs,center,k):
    """
    This function projects vectors within an epsilon ball onto a hyperplane defined be given eigenvectors.

    Parameters:
        vectors (arr): The set of points (vectors) to be projected onto the hyperplane
        eigvecs (arr): The set of eigenvectors which create the hyperplane
        center (arr): The center of both the epsilon ball and the hyperplane
        k (int): The intrinsic dimensions of the hyperplane
    """
    
    newvecs = vectors-center
    projections_list = [] # creates an empty array as a projection
    for i in range(np.shape(vectors)[0]):
        for j in range(k):
            projections_list.append(np.dot(newvecs[i,:k],eigvecs[j])*eigvecs[j])
        
    return(np.sum(projections_list))
