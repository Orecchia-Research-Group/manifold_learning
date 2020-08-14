#Import Packages
import numpy as np

#Import MLS
from mls.moving_least_squares import weight, weight_scaled, dweight, dweight_scaled, ddweight, ddweight_scaled, C_1_MLS_oracle

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
 
    for i in radii:        
        for j in range(len(sorted_vec)):
            if (sorted_vec[j] <= i) and ((sorted_vec[j] > radii[radii.index(i)-1]) or (radii.index(i) == 0)) :
                X.append(cloud[indices[j], :])
        X_mat = np.vstack(X)  # creates a 'matrix' by vertically stacking the elements of the list
        dim_X = np.shape(X_mat)  # saves dimensions of matrix for points within the current radius
        shapes.append(dim_X)
        
        tuples = []
        t_elements = [i]

        if radii.index(i) == 0:
            cov_X = np.cov(X_mat, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov_X)  # computes the eigenvalues and eigenvectors of the covariance matrix
            eigval_list.append(eigvals)  # appends the set of eigenvalues to the list created above
            top_eigvecs.append(eigvecs[0:k])
            for eig in eigvals:
                tuples.append((i,eig))            
            
        elif shapes[radii.index(i)] != shapes[radii.index(i)-1]:
            cov_X = np.cov(X_mat, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov_X)  # computes the eigenvalues and eigenvectors of the covariance matr$
            eigval_list.append(eigvals)  # appends the set of eigenvalues to the list created above
            top_eigvecs.append(eigvecs[0:k])
            for eig in eigvals:     
                tuples.append((i,eig))

        else:
            eigval_list.append(eigval_list[-1])
            top_eigvecs.append(top_eigvecs[-1])
            t_elements.extend(eigval_list[-1])
            for eig in eigval_list[-1]:     
                tuples.append((i,eig))


        # Create instance of MLS class, otherwise add tuples
        if radii.index(i) == 0:
            MLS = C_1_MLS_oracle(tuples, 50, 2)
        else:
            MLS.insert(tuple(t_elements))

        ### Start if statement for MLS here (within radii for loop)... List of tuples stored in 'tuples' variable
        if MLS.eval(i)[1] <= 0:
            break


