import numpy as np
from pymanopt.manifolds import Sphere
import mala.potentials
import mala.metropolis_hastings as mh
import mala.icosehedron as ico
import networkx as nx


def test_chart_maps():
    G = ico.generate_G(verbose=False)
    V = ico.generate_V(G)


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

        # our our centroid and vertices should be mapped to wihtin our chart radius
        for point in face_obj.list_vert_coors()+[face_obj.ctd_coors]:
            assert ico.check_if_point_in_chart(point,face_obj)

        # our centroid should be in our face
        assert np.all(ico.check_if_point_in_face(face_obj.ctd_coors,face_obj))
                        
        # our vertices should be JUST too far out, so we'll pull them towards
        # the centroid and then they should be within our char
        for point in face_obj.list_vert_coors():
            centered_pt = 0.1*(face_obj.ctd_coors - point) + 0.9*point
            assert np.all(ico.check_if_point_in_face(centered_pt,face_obj))

        # make sure we preserve our point when movie back and forth
        test_pt_chart = np.random.normal(loc=1,scale=0.1,size=2)
        test_pt_euclidean = face_obj.chart2euclidean(test_pt_chart)
        assert np.all(np.isclose(test_pt_chart,
                face_obj.euclidean2chart(test_pt_euclidean)))

            
def test_transitions():
    face_graph,vertex_graph,face_dict = ico.generate_icosahedron()

    # for each edge in our graph, make sure face_across_edge recovers
    # the right face
    for edge in face_graph.edges():
        origin_node = edge[0]
        destination_node = edge[1]

        # figure out which edge of our origin face we've crossed
        edge_vertices = list(set(origin_node) & set(destination_node))
        ID_verts = [(v.name in edge_vertices) for
            v in face_dict[origin_node].list_vert_objs()]
        assert np.sum(ID_verts)==2
        # build a list of booleans showing whether we've crossed l1,l2, or l3
        # l1 if verts 2 and 3, l2 if verts 1 and 3, l3 if verts 1 and 2
        side_crossings = [(ID_verts[1] and ID_verts[2]),
                            (ID_verts[0] and ID_verts[2]),
                            (ID_verts[0] and ID_verts[1]) ]
        # retrieve the face predicted by face_across_edge
        predicted_face_obj = ico.face_across_edge(face_dict[origin_node],
                                side_crossings,face_graph)

        assert set(predicted_face_obj)==set(destination_node)









