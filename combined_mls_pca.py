#Import Packages
import numpy as np

def mls_pca(cloud, center_ind, k, radint = .01):
    """
    This function performs PCA and MLS at increasing radii values of an epsilon ball.
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

    

