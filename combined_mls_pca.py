#Import Packages
import numpy as np

#Import MLS
from mls.moving_least_squares import weight, weight_scaled, dweight, dweight_scaled, ddweight, ddweight_scaled, C_1_MLS_oracle

def mls_pca(cloud, center_ind, k, radint = .01, iter=False, dist=None):
    """
    This function performs PCA and MLS at increasing radii values of an epsilon ball.
    """
    
    dim_array = np.shape(cloud)  # saves the dimensions of the array
    eigval_list = []  # creates empty list to store eigenvalues
    top_eigvecs = []  # creates empyt list in order to store the egenvectors of the intrinsic dimension
    if dist == None:
        dist_mat = np.zeros((dim_array[0],dim_array[0])) # creates an empty n x n matrix
    else:
        dist_mat = dist

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

    #value to use for MLS condition
    comp = k*np.log(k)

    # Conditional variable to track if R_min has been identified
    min_found = 0
    max_found = 0

    #Setting R_min to be huge number in order for loop not to cut out early
    R_min = 2**10
    R_max = 0
 
    for i in range(len(radii)):   
         
        for j in range(len(sorted_vec)):
            if (sorted_vec[j] <= radii[i]) and ((sorted_vec[j] > radii[i-1]) or (i == 0)) :
                X.append(cloud[indices[j], :])
        X_mat = np.vstack(X)  # creates a 'matrix' by vertically stacking the elements of the list
        dim_X = np.shape(X_mat)  # saves dimensions of matrix for points within the current radius
        shapes.append(dim_X)

        # Second value to compare with mls condition
        ball = len(X)
        
        if i == 0:
            cov_X = np.cov(X_mat, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov_X)  # computes the eigenvalues and eigenvectors of the covariance matrix
            eigval_list.append(eigvals)  # appends the set of eigenvalues to the list created above
            top_eigvecs.append(eigvecs[0:k])
            
        elif shapes[i] != shapes[i-1]:
            cov_X = np.cov(X_mat, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov_X)  # computes the eigenvalues and eigenvectors of the covariance matr$
            eigval_list.append(eigvals)  # appends the set of eigenvalues to the list created above
            top_eigvecs.append(eigvecs[0:k])

        else:
            eigval_list.append(eigval_list[-1])
            top_eigvecs.append(top_eigvecs[-1])
            eigvals = eigval_list[-1]     
 
        # Set up the list of radii from the tuple (eigenvalues list is already saved as eigvals)
        rad_list = [radii[i]]
        eigvals = np.flip(eigvals)

        eigval_top = [eigvals[0]]
        eigval_k_1 = [eigvals[k]]

        # Set up arrays to pass to MLS
        pairs_top = np.array([rad_list, eigval_top]) 
        pairs_k_1 = np.array([rad_list, eigval_k_1])


        # Create instance of MLS class, otherwise add tuples
        if i == 0:
            MLS_1 = C_1_MLS_oracle(pairs_top, 500, 5)
            MLS_k_1 = C_1_MLS_oracle(pairs_k_1, 500, 5)
        elif i == 1 or i == 2:
            MLS_1.insert(pairs_top)
            MLS_k_1.insert(pairs_k_1)
        else:
            MLS_1.insert(pairs_top)
            MLS_k_1.insert(pairs_k_1)

            ### Start if statement for MLS here (within radii for loop)
            if (abs(MLS_1.eval(radii[i])[1]) <= 0.5) and MLS_1.eval(radii[i])[1]<0 and min_found==1 and max_found == 0:
                R_max = radii[i]
                max_found = 1
                if not(iter):
                    break

            if (MLS_k_1.eval(radii[i])[1]<-0.01) and min_found == 0 and ball >= comp:
                R_min = radii[i]
                min_found = 1
        

    new_radii = radii[:len(eigval_list)]

    if len(eigval_list) == 0:
        raise ValueError(str(eigval_cache) + '\n\n\n' + str(top_eigvecs) + "\n\n\n" + str(radii))
    print(R_min)
    print(R_max)
    return[np.array(eigval_list),np.array(top_eigvecs),new_radii, R_min, R_max]
