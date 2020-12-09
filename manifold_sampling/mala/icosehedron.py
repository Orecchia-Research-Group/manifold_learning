
import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.stats

from mala.potentials import to_spherical_coors


# USEFUL CONSTANTS--------------------------------------------------------------
phi = (1 + 5 ** 0.5) / 2

# Group generators
# order 3
G_1 = np.array([
    [0,0,1],
    [1,0,0],
    [0,1,0]
])
# order 2
G_2 = np.diag([-1,1,1])

# MATRIX GROUP -----------------------------------------------------------------

class icosehedron_rotation:
    def __init__(self,matrix_rep,name):
        self.mat = matrix_rep
        self.name = name

# some wordy code for brute-force showing all elements of group G
def generate_G(**kwargs):
    if 'verbose' in kwargs and kwargs['verbose']:
        verbose = True
        print('Adding elements:')
    else:
        verbose = False

    G_elts = list()

    gen_names = ['G_1','G_2']

    # consider candidates
    diag_sign_matrices = [ G_1@G_1@G_1,G_2, G_1@G_2@G_1@G_1, G_1@G_1@G_2@G_1,
                        G_1@G_2@G_1@G_2@G_1, G_1@G_1@G_2@G_1@G_2,
                        G_1@G_2@G_1@G_1@G_2, G_1@G_2@G_1@G_2@G_1@G_2]
    dsm_names = ['Identity','G_2','G_1@G_2@G_1^2','G_1^2@G_2@G_1',
                        'G_1@G_2@G_1@G_2@G_1','G_1^2@G_2@G_1@G_2',
                        'G_1@G_2@G_1@G_1@G_2','G_1@G_2@G_1@G_2@G_1@G_2']
    for postappend_G_1 in range(3):
        for idx,M in enumerate(diag_sign_matrices):
            if postappend_G_1==0:
                    name = dsm_names[idx]
            else:
                name = 'G_1^'+str(postappend_G_1)+'@'+dsm_names[idx]
            M = np.linalg.matrix_power(G_1,postappend_G_1)@M
            group_elt = icosehedron_rotation(M,name)
            G_elts.append(group_elt)
    if verbose:     
        print('Length is ',len(G_elts))
        for M in G_elts:
            print(M.name,':')
            print(M.mat)
            res = list(np.array_equal(M.mat, M_i.mat) for M_i in G_elts)
            print('     Appears x',np.sum(res))
    return G_elts


# VERTICES AND CENTROIDS--------------------------------------------------------

class vertex:
    def __init__(self,p,name):
        self.p = p
        self.name = name

def generate_V(G):
    v_0 = np.array([1,phi,0])

    V = list()
    for M in G:
        V.append(vertex(np.matmul(M.mat,v_0),M.name+'*v_0'))
    # add last bc when removing duplicates FIRST instance will be removed
    V.append(vertex(v_0,'v_0'))
    # remove duplicates
    while len(V)!=12:
        for idx,u in enumerate(V):
            if np.sum(np.all(np.isclose(u.p, v.p)) for v in V)>1:
                del V[idx]

    return V

# FACES ------------------------------------------------------------------------

# the coordinates and spanning vectors of an icosehedron face in R^3
class ambient_face:
    def __init__(self,v_1,v_2,v_3):
        self.v_1 = v_1
        self.v_2 = v_2
        self.v_3 = v_3

        self.ctd_coors = np.mean([v_1.p,v_2.p,v_3.p],axis=0)

        # we choose our basis vectors canonically so that b_1 is parallel to
        # the line from v_1 to v_2, and so that b_2 is parallel to the line from
        # the midpoint to v_3
        self.basis_1 = (v_2.p - v_1.p)/np.linalg.norm(v_2.p - v_1.p,ord=2)

        mdpt = np.mean([v_1.p,v_2.p],axis=0)
        self.basis_2 = (v_3.p - mdpt)/np.linalg.norm(v_3.p - mdpt,ord=2)

    def coor_shift_by_g(self,g):
        return [np.matmul(g.mat,self.v_1.p),np.matmul(g.mat,self.v_2.p),
        np.matmul(g.mat,self.v_3.p)]

    def return_coors(self):
        return [self.v_1.p,self.v_2.p,self.v_3.p]


# PLANE PROJECTION -------------------------------------------------------------
# recall command to_spherical_coors(x) in potentials.py

def get_angles(x):
    return to_spherical_coors(x)[1:]

# build 3x3 rotation matrix corresponding to clockwise rotation by theta about z
def R_z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta),0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
        ])

# build 3x3 rotation matrix corresponding to clockwise rotation by theta about y
def R_y(theta):
    return np.array([
        [np.cos(theta), 0 , np.sin(theta)],
        [0 , 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
        ])

# Build the matrix that rotates elts of a face to their 3D position
def chart_transformation(face):
    theta_c, phi_c = get_angles(face.ctd_coors)
    v_prime = np.matmul(R_y(-phi_c)@R_z(-theta_c),face.basis_1)
    theta_v,_ = get_angles(v_prime)
    return R_z(-theta_v) @ R_y(-phi_c) @ R_z(-theta_c)

# x is some numpy array with R^3 coordinates
def euclidean2chart(x,face):
    R = chart_transformation(face)
    return np.matmul(R,x)[:2]

# x is some numpy array with R^2 coordinates
def chart2euclidean(x,face):
    R = chart_transformation(face)
    # Our pre-image of our Euclidean embedding uses our chart coordinates as
    # our x- and y-coors, and then assumes a z-coordinate of 1
    x_augmented = np.array([x[0],x[1],1.0])
    return np.matmul(np.linalg.inv(R),x)


