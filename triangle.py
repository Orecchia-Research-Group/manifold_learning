import numpy as np
import networkx as nx
import matplotlib.pyplot as pyplot

def triangle(num_iterations, points):
    if num_iterations > 0:
        points.extend(triangle(num_iterations - 1, [points[0], midpoint(points[0], points[1]), midpoint(points[0], points[2])]))
        points.extend(triangle(num_iterations - 1, [midpoint(points[0], points[2]), midpoint(points[2], points[1]), points[2]]))
        points.extend(triangle(num_iterations - 1, [midpoint(points[0], points[1]), points[1], midpoint(points[2], points[1])]))
        points.extend(triangle(num_iterations - 1, [midpoint(points[0], points[2]), midpoint(points[2], points[1]), midpoint(points[1], points[0])]))
    return points

def midpoint(point1, point2):
    return ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)

def call_triangle(num_iterations, node1 = (0.0,0.0), node2 = (2.0,0.0), node3 = (1.0,1.0)):
    # Creating simple triangle
    output_points = triangle(num_iterations, [node1, node2, node3])
    final_points = []
    for count, point in enumerate(output_points):
        if count % 3 == 0:
            final_points.append(output_points[count:count + 3])
    final_graph = create_graph(final_points)
    print(final_graph.nodes())
    draw_triangle(final_graph)

def create_graph(points):
    my_graph = nx.Graph()
    counter = 0
    for point_set in points:
        new_graph = nx.Graph()
        new_graph.add_node(0, pos = point_set[0])
        new_graph.add_node(1, pos = point_set[1])
        new_graph.add_node(2, pos = point_set[2])
        new_graph.add_edges_from([(0,1),(1,2),(2,0)])
        my_graph = nx.disjoint_union(new_graph, my_graph)
    return my_graph

def draw_triangle(graph):
    positions = nx.get_node_attributes(graph, 'pos')
    nx.draw(graph, positions)
    pass