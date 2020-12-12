
import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.stats

# POTENTIALS -------------------------------------------------------------------
# define the target function we wish to sample
class Potential():
    def __init__(self,ambient_dim,proper_dim,function_handle,grad_handle):
            self.f = function_handle
            self.g = grad_handle
            self.n = ambient_dim
            self.d = proper_dim

    def eval(self,x):
        return self.f(x)

    def gradient(self,x):
        return self.g(x)

# Simple instances
class Gaussian(Potential):
    def __init__(self,ambient_dim,proper_dim,mu,
        cov):
        super().__init__(ambient_dim,proper_dim,function_handle=None,
            grad_handle=None)
        self.mu = mu
        self.cov = cov
        self.f = (lambda x:
            scipy.stats.multivariate_normal.pdf(x, mean=self.mu, cov=self.cov))
        self.g = (lambda x:
            -self.eval(x)*np.matmul(np.linalg.inv(self.cov),x-self.mu))
        self.name = 'Gaussian( '+str(mu)+' , '+str(cov)+' )'

# WARNING: SOMETHING IS WRONG WITH THE GRADIENT CALCULATION OF THIS POTENTIAL.
# gauss_i is a tuple (mu_i, cov_i)
class Double_well(Potential):
    def __init__(self,ambient_dim,proper_dim,gauss_1,gauss_2):
        super().__init__(ambient_dim,proper_dim,function_handle=None,
            grad_handle=None)
        self.mu_1 = gauss_1[0]
        self.cov_1 = gauss_1[1]
        self.mu_2 = gauss_2[0]
        self.cov_2 = gauss_2[1]

        self.f_1 = lambda x: scipy.stats.multivariate_normal.pdf(x, mean=self.mu_1, cov=self.cov_1)
        self.f_2 = lambda x: scipy.stats.multivariate_normal.pdf(x, mean=self.mu_2, cov=self.cov_2)
        self.f = lambda x: self.f_1(x) + self.f_2(x)
        self.g = (lambda x:
            -self.f_1(x)*np.matmul(np.linalg.inv(self.cov_1),x-self.mu_1)
            -self.f_2(x)*np.matmul(np.linalg.inv(self.cov_2),x-self.mu_2) )
        self.name = 'Double well'

class Spherical(Potential):
    def __init__(self,ambient_dim,proper_dim):
        super().__init__(ambient_dim,proper_dim,function_handle=None,
            grad_handle=None)
        # f = ||theta,phi||^2
        self.f = (lambda x: np.linalg.norm(to_spherical_coors(x)[1:]))
        self.g = (lambda x: spherical_gradient(x))
        self.name = 'Spherical'

# Spherical coordinates follow mathematical conventions:
# azimuthal = theta, inclination/"angle from z" = phi
def to_spherical_coors(x):
    r = np.linalg.norm(x,ord=2)

    y = x[1]
    z = x[2]
    x = x[0]

    theta=np.arctan2(y,x)
    phi = np.arccos(z/r)

    """if x==0:
                    theta = np.sign(y)*np.pi/2
                else:
                    theta=np.arctan2(y,x)
                if z==0:
                    phi = np.pi/2
                else:
                    phi = np.arccos(z/r)"""
        #phi =np.arctan(np.sqrt(np.power(x,2)+np.power(y,2))/z)

    return np.array([r,theta,phi])

def from_spherical_coors(x):
    r = x[0]
    theta = x[1]
    phi = x[2]

    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)

    return np.array([x,y,z])

def spherical_gradient(x):
    x_hat = to_spherical_coors(x)
    r = x[0]
    theta = x[1]
    phi = x[2]

    y = x[1]
    z = x[2]
    x = x[0]

    # a dumb amount of chain rule
    par_theta_par_x = (1/(1+np.power(y/x,2)))*(-y/np.power(x,2))
    par_theta_par_y = (1/(1+np.power(y/x,2)))*(1/x)

    v = np.sqrt(np.power(x,2)+np.power(y,2))/z
    v_1 = (1/(1+np.power(v,2)))*(1/(2*z))*np.power(np.power(x,2)+np.power(y,2),
        -0.5)
    par_phi_par_x = v_1*2*x
    par_phi_par_y = v_1*2*y

    par_phi_par_z = -(1/(1+np.power(v,2)))*np.sqrt(np.power(x,2)+np.power(y,
        2))/np.power(z,2)

    dF_dx = 2*theta*par_theta_par_x+2*phi*par_phi_par_x
    dF_dy = 2*theta*par_theta_par_y+2*phi*par_phi_par_y
    dF_dz = 2*phi*par_phi_par_z

    return np.array([dF_dx,dF_dy,dF_dz])


    
