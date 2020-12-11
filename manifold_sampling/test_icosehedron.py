import numpy as np
from pymanopt.manifolds import Sphere
import mala.potentials
import mala.metropolis_hastings as mh
import mala.icosehedron as ico
import networkx as nx


def test_chart_maps():
    G = ico.generate_G(verbose=False)
    V = ico.generate_V(G)

    failures = list()

    face_graph,vertex_graph = ico.build_face_graph(V,verify=True)

    # face dict maps nodes in our face graph to ambient_face objects,
    # which contain geometric information about the face
    face_dict = ico.build_face_map(face_graph,V)

    for face_node in face_graph.nodes():
        face_obj = face_dict[face_node]

        # transform our face under our chart
        transformed_face = face_obj.chart_transform_face()

        # check that our centroid lies on the z-axis
        assert np.all(np.isclose(transformed_face.ctd_coors[:2],[0,0]))
        # check that righthanded faces rotate up, lefthand faces rotate down
        if ico.righthand_face(face_obj):
            assert transformed_face.ctd_coors[2] > 0
        else:
            assert transformed_face.ctd_coors[2] < 0

        # check that our spanning vectors align with x and y
        assert np.all(np.isclose(transformed_face.basis_1,[1,0,0]))

        # our centroid and vertices should all be mapped to wihtin our chart
        # radius
        for point in [face_obj.ctd_coors]+face_obj.list_vert_coors():
            assert ico.check_if_point_in_chart(point,face_obj)