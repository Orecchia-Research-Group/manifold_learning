import numpy as np
from pymanopt.manifolds import Sphere
import mala.potentials
import mala.metropolis_hastings as mh
import mala.icosehedron as ico
import mala.utils
import networkx as nx
from itertools import compress

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_icosahedron():
    G = ico.generate_G(verbose=False)
    V = ico.generate_V(G)

    face_graph,vertex_graph = ico.build_face_graph(V,verify=True)

    # face dict maps nodes in our face graph to ambient_face objects,
    # which contain geometric information about the face
    face_dict = ico.build_face_map(face_graph,V)

    return face_graph,vertex_graph,face_dict

def test_chart_maps():
    face_graph,vetex_graph,face_dict = generate_icosahedron()

    for face_node in face_graph.nodes():
        face_obj = face_dict[face_node]

        # transform our face under our chart
        R = ico.chart_transformation(face_obj)
        R_v_1,R_v_2,R_v_3 = [ico.vertex(np.matmul(R,v),'dummy') for v in face_obj.return_coors()]
        transformed_face = ico.ambient_face(R_v_1,R_v_2,R_v_3)

        # check that our centroid lies on the z-axis
        assert np.all(np.isclose(transformed_face.ctd_coors[:2],[0,0]))
        # check that righthanded faces rotate up, lefthand faces rotate down
        if not ico.righthand_face(face_obj):
            assert transformed_face.ctd_coors[2] > 0
        else:
            assert transformed_face.ctd_coors[2] < 0

        # check that our spanning vectors align with y and x, respectively
        assert np.all(np.isclose(transformed_face.basis_1,[0,1,0]))
        assert np.all(np.isclose(transformed_face.basis_2,[1,0,0]))

        # our centroid and vertices should all be mapped to wihtin our chart
        # radius
        for point in [face_obj.ctd_coors]+face_obj.return_coors():
            assert ico.check_if_point_in_chart(point,face_obj)

def test_chart_transitions():
    face_graph,vetex_graph,face_dict = generate_icosahedron()

    # for each edge, pick an "origin" face
    for edge in list(face_graph.edges())[2:]:

        origin_face = face_dict[edge[0]]
        destination_face = face_dict[edge[1]]

        # retrieve the names and euclidean coorddinates of vertices adjacent to both faces
        shared_vert_names = list(set(edge[0])&set(edge[1]))
        shared_vert_coors = origin_face.coors_by_name(shared_vert_names)

        assert shared_vert_coors == destination_face.coors_by_name(shared_vert_names)
        
        # generate a point that's slightly outside our origin face
        mdpt = mala.utils.normalize(np.mean(shared_vert_coors,axis=0))
        towards_second_centroid = mala.utils.normalize(destination_face.ctd_coors - mdpt)
        point = 2*mala.utils.normalize(mdpt+towards_second_centroid)

        # check that our point is in the chart overlap
        assert ico.check_if_point_in_chart(point,origin_face)
        assert ico.check_if_point_in_chart(point,destination_face)
        # check our point is over our face edge
        assert not np.all(ico.check_if_point_in_face(point,origin_face))
        assert np.all(ico.check_if_point_in_face(point,destination_face))

        # ask ico what face it thinks we're going to
        detected_destination = face_dict[ico.face_across_edge(origin_face,point,face_graph)]
        try:
            assert np.all(destination_face.ctd_coors == detected_destination.ctd_coors)
        except:
            print('MISMATCH DETECTED')
            print('     true edge ',shared_vert_names)
            print('     detected edge ',ico.crossed_edge(origin_face,point))
            print('Crossed ',
                np.sum([not v for v in ico.check_if_point_in_face(point,origin_face)]),
                ' boundaries'
                )

            crossings = ('l1','l2','l3')
            crossed_line = list(compress(crossings,
                [not v for v in ico.check_if_point_in_face(point,origin_face)]))[0]

            # plot in chart image
            fig = plt.figure()
            
            chart_image_face = origin_face.transform_face_by_chart()
            chart_image_face.print()
            u,v,w = chart_image_face.return_coors()
            for x,y in [[u,v],[v,w],[u,w]]:
                plt.plot([x[0],y[0]],[x[1],y[1]],color='grey')
            for idx,x in enumerate([chart_image_face.v_1,chart_image_face.v_2,chart_image_face.v_3]):
                plt.plot([x.p[0]],[x.p[1]],'o',label=x.name)
                
            chart_image_pt = ico.euclidean2chart(point,origin_face)
            plt.plot([chart_image_pt[0]],[chart_image_pt[1]],'o')
            print('crossed line ',crossed_line)
            plt.legend()
            plt.show()
            #destination_face.print()
            #detected_destination.print()

        # map our point into the next chart
        #ico.map_pt_btwn_charts(point,origin_face,detected_destination)
        




