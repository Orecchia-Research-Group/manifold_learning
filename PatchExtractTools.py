# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 21:41:14 2020

@author: FG
"""

import numpy as np
import matplotlib.pyplot as plt
import array
from sklearn.metrics.pairwise import euclidean_distances
from manifold_utils.mSVD import eigen_calc_from_dist_mat

def Dnorm(X):
    '''
    Compute the D-norm of matrix X
    '''
    #Matrix D for computing D-norm
    D = np.array([
        [2, -1, 0, -1, 0, 0, 0, 0, 0],
        [-1, 3, -1, 0, -1, 0, 0, 0, 0],
        [0, -1, 2, 0, 0, -1, 0, 0, 0],
        [-1, 0, 0, 3, -1, 0, -1, 0, 0],
        [0, -1, 0, -1, 4, -1, 0, -1, 0],
        [0, 0, -1, 0, -1, 3, 0, 0, -1],
        [0, 0, 0, -1, 0, 0, 2, -1, 0],
        [0, 0, 0, 0, -1, 0, -1, 3, -1],
        [0, 0, 0, 0, 0, -1, 0, -1, 2]])
    #sqrt(X^T D X)
    Dn = np.sqrt((X.dot(D)*X).sum(axis=1))
    return Dn

def DCT_Transform(b):
    '''
    Transform a vector x in the standard basis into the DCT basis
    '''
    #2-dimensional Discrete Cosine Transform (DCT) basis of a 3×3 image patch diagonalizes the matrix D
    sqrt6 = np.sqrt(6)
    sqrt54 = np.sqrt(54)
    sqrt8 = np.sqrt(8)
    sqrt48 = np.sqrt(48)
    sqrt216 = np.sqrt(216)
    #DCT is 8x9
    DCT = np.array([
        [1/sqrt6, 0, -1/sqrt6, 1/sqrt6, 0, -1/sqrt6, 1/sqrt6, 0, -1/sqrt6 ],
        [1/sqrt6, 1/sqrt6, 1/sqrt6, 0, 0, 0, -1/sqrt6, -1/sqrt6, -1/sqrt6 ],
        [1/sqrt54, -2/sqrt54, 1/sqrt54, 1/sqrt54, -2/sqrt54, 1/sqrt54, 1/sqrt54, -2/sqrt54, 1/sqrt54],
        [1/sqrt54, 1/sqrt54, 1/sqrt54, -2/sqrt54, -2/sqrt54, -2/sqrt54, 1/sqrt54, 1/sqrt54, 1/sqrt54],
        [1/sqrt8, 0, -1/sqrt8, 0, 0, 0, -1/sqrt8, 0, 1/sqrt8],
        [1/sqrt48, 0, -1/sqrt48, -2/sqrt48, 0, 2/sqrt48, 1/sqrt48, 0, -1/sqrt48],
        [1/sqrt48, -2/sqrt48, 1, 0, 0, 0, -1/sqrt48, 2/sqrt48, -1/sqrt48],
        [1/sqrt216, -2/sqrt216, 1/sqrt216, -2/sqrt216, 4/sqrt216, -2/sqrt216, 1/sqrt216, -2/sqrt216, 1/sqrt216]
    ])
    Lambda = np.diag(np.linalg.norm(DCT, ord=2, axis=1)**2)
    #linear transformation
    x = np.dot(np.matmul(Lambda, DCT), b)
    return x

def fnameFix(fn):
    '''
    The numbers in the files are 5 digits.
    Add 0 in front until we have 5 digits.
    '''
    while(len(fn) < 5):
        fn = '0'+ fn
    return fn

def PatExt(fn):
    '''
    Perform patch selections for a single images as detailed in carlsson
    '''
    #dimension of the images
    nrow=1024
    ncol=1536
    #Load the image
    fn = fnameFix(fn)
    filename='iml_Images\imk'+ fn + '.iml'
    s = open(filename, 'rb').read()
    arr = array.array('H', s)
    arr.byteswap()
    img = np.array(arr, dtype='uint16').reshape(nrow, ncol)
    
    #extract 5000 random 3x3 pathches.
    #Note The two leftmost and the two rightmost pixel columns do not contain information
    
    #iniate a matrix to store 5000 flattened 3x3 pathes (5000 samples with 9 features/coordinates)
    Patches = np.zeros((5000,9))
    #the rows and columns we can select from
    rows = np.arange(1, nrow-1)
    cols = np.arange(3, ncol-3)
    for i in range(5000):
        #select a random center
        x = np.random.choice(rows)
        y = np.random.choice(cols)
        #stored the flattened patches
        Patches[i,] = img[x-1:x+2, y-1:y+2].flatten()
    #Compute the logarithm of intensity at each pixel.
    Patches = np.log(Patches)
    #Subtract an average of all coordinates from each coordinate.
    colMeans = np.mean(Patches, axis=0)
    Patches = Patches-colMeans
    #compute D-norms for each vector and record the top 1000 D-norms
    Dns = Dnorm(Patches)
    hC_Indices = np.argsort(Dns)[-1000:]
    #Store the normalized and DCT-transformed high contrast patches
    hC_DCT_Patches = np.zeros((1000,8)) #7-dimensional sphere in R8
    for i in range(1000):
        im_ind = hC_Indices[i] #index of the i-th high contrast patches 
        hC_DCT_Patches[i] = DCT_Transform(Patches[im_ind,]/Dns[im_ind]) #DCT transform the normalized patches
    
    return hC_DCT_Patches

def PatchesExtractions(numImages = 5):
    '''
    Perform patch selections for multiple images
    '''
    #Select numImages amount Images from 4212 images
    img_indices = np.random.choice(np.arange(1, 4213), size=numImages, replace=False)
    Patches = np.zeros((numImages*1000, 8))
    for i in range(numImages):
        fn = str(img_indices[i])
        Patches[i*1000:(i+1)*1000] = PatExt(fn)
    print(img_indices)
    #Patches are 8 dimensional
    #there are numImages*1000 patches, the matrix is numImages*1000 by 8. 
    np.save('3x3Patches', Patches)
    return

def Denoise(k, Patches):
    '''
    Denoise the patches from images using kNN
    k: size of the neighborhood
    '''
    nrow, ncol = Patches.shape
    #create a matrix to stored the Denoised Patches
    Denoised_Patches = np.zeros((nrow, ncol))
    #Considering the rows as vectors, compute the distance matrix between each pair of vectors.
    DistanceMatrix = euclidean_distances(Patches)
    for i in range(nrow):
        #find the indices of the kNN for the i-th data point
        i_kNN_indices = np.argsort(DistanceMatrix[i,])[0:k]
        #Update the value of the i-th data point with the average of its kNN
        Denoised_Patches[i] = np.mean(Patches[i_kNN_indices,], axis=0)
    return Denoised_Patches

def IterativeDenoise(k, fn, ite = 2):
    '''
    Iteratively denoise the patches from images using kNN
    k: size of the neighborhood
    ite: number of iteration
    '''
    Patches = np.load(fn)
    for i in range(ite):
        Patches = Denoise(k, Patches)
    return Patches

'''
#Patches Extractions
PatchesExtractions(numImages = 10)
#Denoise and save
Denoised_Patches = IterativeDenoise(2, '3x3Patches.npy', ite = 2)
np.save('Denoised3x3Patches', Denoised_Patches)
'''

def eigen_plot(eigval_list,radii, R_min, R_max, logscale=False):
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
    if logscale:
        axes.set_yscale('log')
    for i in range(dim_eig_mat[1]):  # iterates through the columns (dimensions) of the eigenvalue matrix
        axes.plot(radii, eig_mat[:, i])  # plots eigenvalues (y-axis) against each radii value (x-axis)
    st = "Manifold_radii" + str(np.random.randint(low=1, high = 1000)) + ".png"
    plt.savefig(st, dpi=400)
    return (plt.show())


def Image_ev_plot(Patches, dist_mat, center_id, log=True):
    radii, eigval_list, _ = eigen_calc_from_dist_mat(Patches, dist_mat, center_id)
    rmin = radii
    rmax = radii
    eigen_plot(eigval_list, radii, rmin, rmax, logscale=log)
    
'''    
#load matrix
Patches = np.load('Denoised3x3Patches.npy')
dist_mat = euclidean_distances(Patches)
#do ev plot around c given by cid
cid = 5000
Image_ev_plot(Patches, dist_mat, cid)
'''













