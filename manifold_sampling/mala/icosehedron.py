import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.stats
import networkx as nx
from itertools import compress

from mala.potentials import to_spherical_coors
import mala.utils

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

# matrices that delineate edge-crossings in our chart image, as per 
# "Isabelle's Icosahedron"
A_1 = np.array([[1,0],
               [0,0]])

A_2 = np.array([[np.cos(2*np.pi/3),-np.sin(2*np.pi/3)],
               [np.sin(2*np.pi/3),np.cos(2*np.pi/3)]])

A_3 = np.array([[np.cos(2*np.pi/3),np.sin(2*np.pi/3)],
               [-np.sin(2*np.pi/3),np.cos(2*np.pi/3)]])

# VERTEX GENERATING GROUP ------------------------------------------------------

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

# ICOSEHEDRON VERTICES ---------------------------------------------------------

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

# ICOSAHEDRON FACES-------------------------------------------------------------

# describes the coordinates and spanning vectors of an icosehedron face in R^3,
# with methods for mapping the face and points into the proper chart
class ambient_face:
    def __init__(self,v_1,v_2,v_3):
        self.v_1 = v_1
        self.v_2 = v_2
        self.v_3 = v_3

        self.ctd_coors = np.mean([v_1.p,v_2.p,v_3.p],axis=0)

        # assume the radius of our chart is twice as big as that of the 
        # faces' circumcircle
        self.circumcircle_radius = np.linalg.norm(v_1.p-self.ctd_coors,ord=2)
        self.chart_radius = 2*self.circumcircle_radius
        self.sphere_radius = np.linalg.norm(self.v_1.p)

        # we choose our basis vectors canonically so that b_1 points from the
        # centroid to v_1, and b_2 is parallel to the line from v_3 to v_2.
        self.basis_1 = mala.utils.normalize(self.v_1.p-self.ctd_coors)
        self.basis_2 = mala.utils.normalize(self.v_3.p - self.v_2.p)

    def coor_shift_by_g(self,g):
        return [np.matmul(g.mat,self.v_1.p),np.matmul(g.mat,self.v_2.p),
        np.matmul(g.mat,self.v_3.p)]

    def list_vert_objs(self):
        return [self.v_1,self.v_2,self.v_3]

    def list_vert_coors(self):
        return [self.v_1.p,self.v_2.p,self.v_3.p]

    def chart_transform_face(self):
        R = chart_transformation(self)
        R_v1,R_v2,R_v3 = [vertex(np.matmul(R,v.p),v.name) 
            for v in self.list_vert_objs()]

        return ambient_face(R_v1,R_v2,R_v3)

    def euclidean2chart(self,x):
        return euclidean2chart(x,self)

    def chart2euclidean(self,x):
        return chart2euclidean(x,self)

    def print(self):
        print('Face centered at ',self.ctd_coors)
        print('     v_1 @ ',np.around(self.v_1.p,decimals=2),' named ',self.v_1.name)
        print('     v_2 @ ',np.around(self.v_2.p,decimals=2),' named ',self.v_2.name)
        print('     v_3 @ ',np.around(self.v_3.p,decimals=2),' named ',self.v_3.name)

    def plot_in_face(self,point,**kwargs):
        if 'in_chart_coors' in kwargs and kwargs['in_chart_coors']:
            transformed_point = point
        else:
            transformed_point = self.euclidean2chart(point)

        # Now plot face and point in the image of the chart
        transformed_face = self.chart_transform_face()
        
        plt.figure()
        # plot triangle sides
        u,v,w = transformed_face.list_vert_coors()
        for x,y in [[u,v],[v,w],[u,w]]:
            plt.plot([x[0],y[0]],[x[1],y[1]],color='grey')
        # scatter and label points
        for idx,x in enumerate([u,v,w]):
            plt.plot([x[0]],[x[1]],'o',label=idx)
        # scatter our point
        plt.plot([transformed_point[0]],[transformed_point[1]],'o',label='point')
        plt.legend()
        plt.show()
        return

def check_if_point_in_chart(p,face,**kwargs):
    # if we're in chart coordinates already, we assume the point is on 
    # same hemisphere as the face, since there's no way to "check"
    if 'in_chart_coors' in kwargs and kwargs['in_chart_coors']:
        p_prime = p
    else:
        # check "facing same way"
        if np.dot(p,face.ctd_coors)<0:
            return False
        # If on same hemisphere, check whether its projection lies within chart 
        # radius
        p_prime = euclidean2chart(p,face)
    return np.linalg.norm(p_prime,ord=2) < face.chart_radius

# return an array indicating if we've crossed the lines l1,l2, or l3
# conventions taken from Isabella's Icosahedron 
def check_if_point_in_face(p,face,**kwargs):
    # if we're in chart coordinates already, we assume the point is on 
    # same hemisphere as the face, since there's no way to "check"
    if 'in_chart_coors' in kwargs and kwargs['in_chart_coors']:
        p_prime = p
    else:
        # check "facing same way"
        if np.dot(p,face.ctd_coors)<0:
            return False
        # If on same hemisphere, check whether its projection lies within chart 
        # radius
        p_prime = euclidean2chart(p,face)
    return [(np.matmul(A,p_prime) > -face.chart_radius/4) 
        for A in [A_1,A_2,A_3]]

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

def righthand_face(face):
    assert np.dot(face.ctd_coors,np.cross(face.basis_1,face.basis_2))!=0
    return np.dot(face.ctd_coors,np.cross(face.basis_1,face.basis_2)) > 0

# Build the matrix that rotates elts of a face to their 3D position
def chart_transformation(face):
    # get angles of centroid
    theta_c, phi_c = get_angles(face.ctd_coors)
    assert theta_c <= np.pi*2
    assert phi_c <= np.pi

    # check "handedness" of face: whether cross-prod of first and 2nd axis points
    # outwards or inwards
    # if right-handed, rotate to top of sphere
    if righthand_face(face):
        v_prime = np.matmul(R_y(-phi_c)@R_z(-theta_c),face.basis_1)
        theta_v,_ = get_angles(v_prime)
        return R_z(-theta_v) @ R_y(-phi_c) @ R_z(-theta_c)
    # if left-handed, flip orientation to align first axis with positive x-axis
    else:
        v_prime = np.matmul(R_y(np.pi-phi_c)@R_z(-theta_c),face.basis_1)
        theta_v,_ = get_angles(v_prime)
        return R_z(-theta_v) @ R_y(np.pi-phi_c) @ R_z(-theta_c)

# x is some numpy array with R^3 coordinates
def euclidean2chart(x,face):
    R = chart_transformation(face)
    return np.matmul(R,x)[:2]

# x is some numpy array with R^2 coordinates
def chart2euclidean(x,face):
    R = chart_transformation(face)
    # Our pre-image of our Euclidean embedding uses our chart coordinates as
    # our x- and y-coors, and then assumes a z-coordinate of 1 or -1 depending
    # on handedness
    if righthand_face(face):
        x_augmented = np.array([x[0],x[1],face.sphere_radius])
    else:
        x_augmented = np.array([x[0],x[1],-face.sphere_radius])
    return np.matmul(np.linalg.inv(R),x_augmented)

# GRAPH REPRESENTATION ---------------------------------------------------------
# We describe our icosahedron using three objects. We build a list of
# vertices v with coordinates in R^3 and names according to the group element 
# that maps v_0 to v. (1) We make a VERTEX GRAPH, where nodes correspond to
# vertices of the icosahedron and are named using the vertices' names, and
# edges correspond to edges of the icosahedron. (2) We make a FACE GRAPH, where
# nodes are tuples of vertex names describing a single face, and edges indicate 
# faces that are adjacent by sharing two vertices. (3) We make a FACE DICT, 
# which maps nodes in the face graph to ambient_face objects which contain 
# geometric information.

def generate_icosahedron():
    G = generate_G(verbose=False)
    V = generate_V(G)

    face_graph,vertex_graph = build_face_graph(V,verify=True)
    face_dict = build_face_map(face_graph,V)

    return face_graph,vertex_graph,face_dict

# given a list of named vertices, builds the face graph, where faces are labeled
# by tuples of the names of their vertices
def build_face_graph(V,**kwargs):
    # first, build the icosahedral graph (nodes correspond to verticess)
    vertex_graph = nx.Graph()
    vertex_graph.add_nodes_from([v.name for v in V])

    for v in V:
        vertex_graph.add_edges_from([(v.name,w.name) 
            for w in five_closest_tuples(v,V)])

    # Then take the dual graph
    face_graph = build_dual_graph(vertex_graph)

    # check that we've built something isomorphic to the true icosehedron 
    # face graph
    if 'verify' in kwargs and kwargs['verify']:
        true_ico = nx.icosahedral_graph()
        true_dual = build_dual_graph(true_ico)

        mapping = dict()
        for idx,v in enumerate(V):
            mapping[v.name]= idx
        relabeled_verts = nx.relabel.relabel_nodes(vertex_graph,mapping,
            copy=True)

        verify_graph = build_dual_graph(relabeled_verts)
        assert nx.is_isomorphic(true_dual,verify_graph)
    
    return face_graph,vertex_graph

# find neighboring vertices. Very crude rn: uses euclidean distance
def five_closest_tuples(target,V_list):
    closest = np.argsort([np.linalg.norm(target.p - v.p,ord=2) for v in V_list])
    # omit closest: will be self
    return [V_list[ii] for ii in closest[1:6]]

# given a networkx graph, constructs a dual graph whose nodes correspond to
# triangular faces
def build_dual_graph(graph,**kwargs):
    # get list of all of the triangles (faces) in icosehedron
    faces=[tuple(x) for x in list(nx.enumerate_all_cliques(graph)) if len(x)==3]

    face_graph = nx.Graph()
    face_graph.add_nodes_from(faces)

    for f in face_graph:
        face_graph.add_edges_from(neighbor_tuples(f,face_graph))

    if 'labels' in kwargs and kwargss['labels']:
        nx.draw(face_graph,with_labels=True)
    
    return face_graph

# given a tuple identifying a face and an ambient graph, identify all faces 
# adjacent to our target 
def neighbor_tuples(target_face,face_graph):
    face_set = set(target_face)
    # two faces are adjacent iff they share two nodes. not this excludes
    # self-loops
    return [(target_face,v) for v in face_graph if len(face_set & set(v))==2]

# build a dictionary mapping nodes in our graph to face objects
def build_face_map(face_graph,V):
    face_map = dict()
    for _,face_tuple in enumerate(face_graph.nodes()):
        verts = [v for v in V if v.name in list(face_tuple)]
        assert len(verts)==3
        candidate_face = ambient_face(verts[0],verts[1],verts[2])
        # If not righthanded, swap two vertices to make it righthanded
        if not righthand_face(candidate_face):
            candidate_face = ambient_face(verts[1],verts[0],verts[2])
        face_map[face_tuple] = candidate_face
    return face_map

def face_across_edge(face,side_crossings,face_graph):
    """
    side_crossings is a list of booleans indicating whether we're crossing
    over l1, l2, or l3 (recall these correspond to the lines between 
    [ (x2,x3),(x1,x3),or (x1,x2)] respectively), as per conventions
    in "Isabelle's Icosahedron"

    NOTE: This implementation is correct so long as there is exactly one
    True in side_crossings. This method could be made faster by only searching
    faces adjacent to the original face in face_graph
    """
    assert np.sum(side_crossings)==1
    # first, ID the edge we're traversing using a pair of vertices
    face_verts = (face.v_1.name,face.v_2.name,face.v_3.name)
    vertex_pairs = [(face_verts[1],face_verts[2]),(face_verts[0],face_verts[2]),
                    (face_verts[0],face_verts[1])]
    edge = list(compress(vertex_pairs,side_crossings))[0]

    # find a face that contains the same edge that isn't our og face
    edge_adjacent_face = [f for f in face_graph if len(set(edge) & set(f))==2
                       and set(f)!=set(face_verts)]
    # there should only be one face that meets these criteria
    assert len(edge_adjacent_face)==1
    return edge_adjacent_face[0]

def map_pt_btwn_charts(pt,origin_face,dest_face):
    """
    pt is a numpy array of coordinates in the image of our origin face chart.
    pt needs to be in the overlap of the face charts
    """
    try:
        assert check_if_point_in_chart(pt,origin_face,in_chart_coors=True)
    except:
        print(pt)
    assert np.all(check_if_point_in_face(origin_face.chart2euclidean(pt),
        dest_face,in_chart_coors=False))

    # first, get Euclidean coordinates of our point
    euc_pt = chart2euclidean(pt,origin_face)

    # next, move into coordinates of second chart
    return euclidean2chart(euc_pt,dest_face)



