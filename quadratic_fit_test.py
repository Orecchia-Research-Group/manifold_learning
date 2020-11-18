from manifold_utils.mSVD import hypersphere
#from combined_mls_pca import mls_pca 
from fit_quad_diff import manifold_regression as mr
import numpy as np
import matplotlib.pyplot as plt

## Sample within unit circle R^2
length = np.sqrt(np.random.uniform(0,1,1000))
#print(length)

theta = np.pi * np.random.uniform(0,2,1000)
#print(theta)

x=length * np.cos(theta)
y=length * np.sin(theta)
z=0.5*x**2-3.0*y**2

#plt.scatter(x,y)
#plt.show()

V=np.vstack([[1,0],[0,1],[0,0]])
V_perp=np.vstack([[0],[0],[1]])

table=np.stack((x,y,z),axis=1)
#cov_X = np.cov(table, rowvar=False)
#eigvals, eigvecs = np.linalg.eigh(cov_X)  # computes the eigenvalues and eigenvectors of the covariance matrix

#eigvecs=np.vstack(eigvecs)
#top_eig = eigvecs[:,:2]
#bot_eig = eigvecs[:,-1]

#print(top_eig)
#print(np.shape(top_eig))
#print(bot_eig)

#coef = mr(top_eig,bot_eig,table,[0,0,0])
coef = mr(V,V_perp,table,[0,0,0])
print(coef)

#print(coef)
#print(table)


### TEST USING HYPERSPHERE AND MLS_PCA FUNCTIONS

#hypersphere = hypersphere(1000,3) #Create S^2
#hypersphere = np.stack(hypersphere,axis=1)
#print(hypersphere)
#print(np.shape(hypersphere))

#center = hypersphere[500,:]
#eigvals,top_eigvecs,radii,R_min, R_max, points, bottom_eigvecs = mls_pca(hypersphere,500,2) #Perform mls_pca on the hypersphere
#print(np.shape(points))
#print(top_eigvecs)
#print(np.shape(top_eigvecs))
#print(np.shape(bottom_eigvecs))
#print(R_min,R_max)


#eig_ind=np.shape(top_eigvecs)[0]//2
#print(eig_ind)

#new_rad=eig_ind*.01+R_min


#V=top_eigvecs[eig_ind]
#print(np.shape(V))
#V_perp=bottom_eigvecs[eig_ind]
#print(np.shape(V_perp))

# Grab correct points
#dim_array = np.shape(hypersphere)  # saves the dimensions of the array

#dist_mat = np.zeros((dim_array[0],dim_array[0]))

# Fill in empty matrix with distances from each point to each other
#for i in range(dim_array[0]):
#    for j in range(dim_array[0]):
#        dist_mat[i,j] = np.linalg.norm(hypersphere[i,:]-hypersphere[j,:])

# Select the distance vector to work with based on the center_ind
#dist_vec = dist_mat[500,:].copy()
#sorted_vec = np.sort(dist_vec)
#indices = [*range(len(sorted_vec))]
#indices.sort(key=lambda x: dist_vec[x]) # sorts indices (of original points)
#shapes = [] # creates empty list to store shapes of X
#X = []

#for j in range(len(sorted_vec)):
#    if (sorted_vec[j] <= new_rad) :
#        X.append(hypersphere[indices[j], :])
#X_mat = np.vstack(X)  # creates a 'matrix' by vertically stacking the elements of the list


#coef=mr(V.T,V_perp.T,X_mat,center)
#print(coef)
