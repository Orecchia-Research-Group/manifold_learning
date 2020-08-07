# Project Introduction 

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
