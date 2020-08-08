# Project Introduction 

Manifold learning is a subfield of machine learning that seeks to leverage the following observation: That empirical probability distributions in various sciences tend to possess an underlying, low-dimensional structure. Manifold learning methods are both ubiquitous and diverse; principal components analysis ([PCA](https://en.wikipedia.org/wiki/Principal_component_analysis)), a staple method in dimensionality reduction and linear modeling, assumes that a probability distribution is well approximated by a low-dimensional Gaussian distribution. More generally applical methods, such as t-distributed stochastic neighbor embedding ([tSNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)) and uniform manifold approximation for dimension reduction ([UMAP](https://umap-learn.readthedocs.io/en/latest/)), seek to capture features of probability distributions whose underlying manifolds are highly nonlinear---at the expense of the preservation of locally-affine structure exhibited by PCA. To our knowledge, no manifold learning method can create low-dimensional manifold representations of point clouds that both 1) preserve topological information for arbitrary manifold, and 2) allow for computation within the manifold in terms of a (global) atlas of coordinate charts (this is the way geometers tend to perform manifold computations; for example, see Guggenheimer's _Differential Geometry_).

Little _et al._ (2017) present multiscale singular value decomposition (mSVD): a method which allows one to estimate the intrinsic dimensionality of a (generally varifold but potentially merely mani)fold structure underlying a point cloud at a fixed point (i.e. locally). Simultaneously, this method gives a way of computing a hyperplane that lays close (in the [Grassmannian](https://en.wikipedia.org/wiki/Grassmannian) sense) to the actual tangent hyperplane of the underlying manifold. This methodology, we believe, serves as a sturdy foundation upon which to build a manifold learning technique that creates an atlas for intrinsic computation within an underlying manifold.

This repository seeks to build a library that 1) impliments Little's mSVD at local points in a point cloud; 2) uses local hyperplane approximations output by mSVD to stitch together an atlas of coordinate charts that cover the entire manifold (defining "entire" to mean "so far as is represented by the point cloud"). Our current iteration of the project splits the mSVD procedure into two portions: the eigendecomposition of balls of increasing radii with fixed center within a point cloud, and the approximation of zeroth and first derivatives of the corresponding eigenvalues as a function of radius. This latter portion is done through Moving Least Squares approximation.

# Moving Least Squares

In order to approximate changes in eigenvalues over different radii, the Moving Least Squares algorithm was implemented (as seen in [Scattered Data Approximation](https://www.cambridge.org/core/books/scattered-data-approximation/980EEC9DBC4CAA711D089187818135E3) by Holger Wendland) for 2D functions in the `C_1_MLS_oracle` class of `moving_least_squares.py`. 

The power of Moving Least Squares lies in its ability to generate approximations for individual input values, rather than computing one global approximation. 
A weight function (`weight()` and first derivative `dweight()` in `moving_least_squares.py`) governs which points are considered in approximating a local input value `x`. 
A scaling factor of `delta` allows the user to refine the range of points considered in each approximation; larger values of delta usually improve the quality of the approximation. 
The value chosen for delta is then passed to a scaled version of the weight function `weight_scaled()` and its first derivative `dweight_scaled()`.

To use Moving Least Squares, a 2D array of coordinates (`points`), scaling factor `delta`, and degree of approximation `m` are passed as input to create an instance of the class `C_1_MLS_oracle`.
Then, the function `eval(x)` can be called for a local input value `x` and will return the Moving Least Squares approximation of `x` and its first derivative.

The `test_weight_ftn.py` and `test_MLS.py` files test the functionality of the weight function and Moving Least Squares approximation, respectively. 
To run these tests, the user can simply call `nosetests` or `nosetests3` from the `tests` directory.
Additionally, the notebook `MLS_Plotting.ipynb` provides a visual representation of the power of Moving Least Squares.
