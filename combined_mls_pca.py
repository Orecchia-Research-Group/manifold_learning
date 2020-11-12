#Import Packages
import numpy as np

# Get handle on NoneType
NoneType = type(None)

#Import MLS
from mls.moving_least_squares import weight, weight_scaled, dweight, dweight_scaled, ddweight, ddweight_scaled, C_1_MLS_oracle

#Import two_index_iterator
from manifold_utils.mSVD import two_index_iterator

def mls_pca(cloud, center_ind, k, radint = .01, iter=False, dist=None):
    """
    This function performs PCA and MLS at increasing radii values of an epsilon ball.
    """
    # Calculates step value k and delta for MLS
    delta = 10*radint
    k_step = np.floor(delta/radint)*radint

    dim_array = np.shape(cloud)  # saves the dimensions of the array
    eigval_list = []  # creates empty list to store eigenvalues
    top_eigvecs = []  # creates empyt list in order to store the egenvectors of the intrinsic dimension
    bottom_eigvecs = []

    if isinstance(dist, NoneType):
        dist_mat = np.zeros((dim_array[0],dim_array[0])) # creates an empty n x n matrix
        # Fill in empty matrix with distances from each point to each other
        for i in range(dim_array[0]):
            for j in range(dim_array[0]):
                dist_mat[i,j] = np.linalg.norm(cloud[i,:]-cloud[j,:])
    else:
        dist_mat = dist

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
    R_max = 2**10

    count = 0
    len_arr = [0]*len(radii)
 
    for i in range(len(radii)):   
         
        for j in range(len(sorted_vec)):
            if (sorted_vec[j] <= radii[i]) and ((sorted_vec[j] > radii[i-1]) or (i == 0)) :
                X.append(cloud[indices[j], :])
        X_mat = np.vstack(X)  # creates a 'matrix' by vertically stacking the elements of the list
        dim_X = np.shape(X_mat)  # saves dimensions of matrix for points within the current radius
        shapes.append(dim_X)

        len_arr[i] = len(X)
        
        if i == 0:
            cov_X = np.cov(X_mat, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov_X)  # computes the eigenvalues and eigenvectors of the covariance matrix
            eigval_list.append(eigvals)  # appends the set of eigenvalues to the list created above
            top_eigvecs.append(eigvecs[0:k])
            bottom_eigvecs.append(eigvecs[k:])
                
        elif shapes[i] != shapes[i-1]:
            cov_X = np.cov(X_mat, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov_X)  # computes the eigenvalues and eigenvectors of the covariance matr$
            eigval_list.append(eigvals)  # appends the set of eigenvalues to the list created above
            top_eigvecs.append(eigvecs[0:k])
            bottom_eigvecs.append(eigvecs[k:])

        else:
            eigval_list.append(eigval_list[-1])
            top_eigvecs.append(top_eigvecs[-1])
            eigvals = eigval_list[-1]
            bottom_eigvecs.append(eigvecs[k:])     
 
        # Set up the list of radii from the tuple (eigenvalues list is already saved as eigvals)
        rad_list = [radii[i]]
        eigvals = np.flip(eigvals)

        eigval_top = [eigvals[0]]
        eigval_k_1 = [eigvals[k]]

        # Set up arrays to pass to MLS
        pairs_top = np.array([rad_list, eigval_top]) 
        pairs_k_1 = np.array([rad_list, eigval_k_1])

        if i == 0:
            MLS_1 = C_1_MLS_oracle(pairs_top, delta, 2)
            MLS_k_1 = C_1_MLS_oracle(pairs_k_1, delta, 2)
            count += 1 
        elif i <= np.floor(delta/radint):
            MLS_1.insert(pairs_top)
            MLS_k_1.insert(pairs_k_1)
            count += 1 
        else:
            MLS_1.insert(pairs_top)
            MLS_k_1.insert(pairs_k_1)
            ball = len_arr[i-count]
           
            ### Start if statement for MLS here (within radii for loop)
            if MLS_1.eval(radii[i] - k_step)[1]>0.01 and R_max != 2**10 and max_found == 0:
                max_found == 1
                if not(iter):
                    break

            if (MLS_1.eval(radii[i] - k_step)[1]< 0 or np.isclose(0, MLS_1.eval(radii[i] - k_step)[1])) and min_found==1 and max_found ==0:
                R_max = radii[i] - k_step


            if (abs(MLS_k_1.eval(radii[i] - k_step)[1]) < 0.1 or MLS_k_1.eval(radii[i] - k_step)[1]<0) and min_found == 0 and ball >= comp:
                R_min = radii[i] - k_step
                min_found = 1

        
    #print(cloud)
    #print(eigval_list)
    #print(cov_X)
    new_radii = radii[:len(eigval_list)]

    if len(eigval_list) == 0:
        raise ValueError(str(eigval_cache) + '\n\n\n' + str(top_eigvecs) + "\n\n\n" + str(radii))
#    print(R_min)
#    print(R_max)
    return[np.array(eigval_list),np.array(top_eigvecs),new_radii, R_min, R_max, X_mat, bottom_eigvecs]

def rapid_mls_pca(cloud, center_ind, k, radint = .01, iter=False, dist=None):
    """
    This function performs PCA and MLS at increasing radii values of an epsilon ball.
    """
    # Calculates step value k and delta for MLS
    delta = 10*radint
    k_step = np.floor(delta/radint)*radint

    dim_array = np.shape(cloud)  # saves the dimensions of the array
    eigval_list = []  # creates empty list to store eigenvalues
    top_eigvecs = []  # creates empyt list in order to store the egenvectors of the intrinsic dimension
    bottom_eigvecs = []

    if isinstance(dist, NoneType):
        dist_mat = np.zeros((dim_array[0],dim_array[0])) # creates an empty n x n matrix
        # Fill in empty matrix with distances from each point to each other
        for i in range(dim_array[0]):
            for j in range(dim_array[0]):
                dist_mat[i,j] = np.linalg.norm(cloud[i,:]-cloud[j,:])
    else:
        dist_mat = dist

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
    R_max = 2**10

    count = 0

#    for i in range(len(radii)):   
#         
#        for j in range(len(sorted_vec)):
#            if (sorted_vec[j] <= radii[i]) and ((sorted_vec[j] > radii[i-1]) or (i == 0)) :
#                X.append(cloud[indices[j], :])
#        X_mat = np.vstack(X)  # creates a 'matrix' by vertically stacking the elements of the list
#        dim_X = np.shape(X_mat)  # saves dimensions of matrix for points within the current radius
#        shapes.append(dim_X)

    for rad, cands in two_index_iterator(radii, indices, key=lambda x: dist_vec[x]):
        if len(cands) > 0:
            new_cands = np.stack([cloud[cand, :] for cand in cands], axis=0)
            try:
                X = np.vstack([X, new_cands])
            except NameError:
                X = new_cands
            cov_X = np.cov(X, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov_X)
            eigval_list.append(eigvals)
            top_eigvecs.append(eigvecs[0:k])
            bottom_eigvecs.append(eigvecs[k:])
        else:
            eigval_list.append(eigval_list[-1])
            top_eigvecs.append(top_eigvecs[-1])
            bottom_eigvecs.append(bottom_eigvecs[-1])

        # Set up the list of radii from the tuple (eigenvalues list is already saved as eigvals)
        rad_list = [rad]
        eigvals = np.flip(eigvals)

        eigval_top = [eigvals[0]]
        eigval_k_1 = [eigvals[k]]

        # Set up arrays to pass to MLS
        pairs_top = np.array([rad_list, eigval_top])
        pairs_k_1 = np.array([rad_list, eigval_k_1])

        try:
            MLS_1.insert(pairs_top)
            MLS_k_1.insert(pairs_k_1)
        except NameError:
            MLS_1 = C_1_MLS_oracle(pairs_top, delta, 2)
            MLS_k_1 = C_1_MLS_oracle(pairs_k_1, delta, 2)

        ### Start if statement for MLS here (within radii for loop)
        if MLS_1.eval(radii[i] - k_step)[1]>0.01 and R_max != 2**10 and max_found == 0:
            max_found == 1
            if not(iter):
                break

        if (MLS_1.eval(radii[i] - k_step)[1]< 0 or np.isclose(0, MLS_1.eval(radii[i] - k_step)[1])) and min_found==1 and max_found ==0:
            R_max = radii[i] - k_step


        if (abs(MLS_k_1.eval(radii[i] - k_step)[1]) < 0.1 or MLS_k_1.eval(radii[i] - k_step)[1]<0) and min_found == 0 and ball >= comp:
            R_min = radii[i] - k_step
            min_found = 1

    new_radii = radii[:len(eigval_list)]

    if len(eigval_list) == 0:
        raise ValueError(str(eigval_cache) + '\n\n\n' + str(top_eigvecs) + "\n\n\n" + str(radii))

    return[np.array(eigval_list), np.array(top_eigvecs), new_radii, R_min, R_max, bottom_eigvecs]
