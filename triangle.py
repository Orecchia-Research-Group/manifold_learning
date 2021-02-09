import numpy as np
import networkx as nx
import matplotlib.pyplot as pyplot

PHI = 1.61803
DIST = 

class icosahedron:
    def __init__(self):
        i_graph = nx.Graph()
        vertices = [(1, PHI, 0), (-1, PHI, 0), (1, -PHI, 0), (-1, -PHI, 0), (0, 1, PHI), (0, 1, -PHI), (0, -1, PHI), (0, -1, -PHI), (PHI, 0, 1), (-PHI, 0, 1), (PHI, 0, -1), (-PHI, 0, -1)]

        vertex_families = {}
        other_vertices = vertices.copy()
        # Create nodes for icosahedron
        count = 0
        for vertex in vertices:
            i_graph.add_node(count, pos = vertex)
            count += 1
            for vertex2 in other_vertices:
                if dist(vertex, vertex2) < DIST: # Need to define 3d Euclidean distance or use buil-in numpy function
                    vertex_families[vertex].append(vertex2)
            other_vertices.remove(vertex)



        

class triangle:
    def __init__(self, num_iterations, node1 = (0.0,0.0, 0.0), node2 = (2.0,0.0, 0.0), node3 = (1.0,1.0, 0.0)):
        output_points = triangle(num_iterations, [node1, node2, node3])
        final_points = []
        for count, point in enumerate(output_points):
            if count % 3 == 0:
                final_points.append(output_points[count:count + 3])
        
        self.points = final_points
        self.graph = create_graph(final_points)

    def triangle(num_iterations, points):
        if num_iterations > 0:
            points.extend(triangle(num_iterations - 1, [points[0], midpoint(points[0], points[1]), midpoint(points[0], points[2])]))
            points.extend(triangle(num_iterations - 1, [midpoint(points[0], points[2]), midpoint(points[2], points[1]), points[2]]))
            points.extend(triangle(num_iterations - 1, [midpoint(points[0], points[1]), points[1], midpoint(points[2], points[1])]))
            points.extend(triangle(num_iterations - 1, [midpoint(points[0], points[2]), midpoint(points[2], points[1]), midpoint(points[1], points[0])]))
        return points

    def midpoint(point1, point2):
        return ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2, (point1[2] + point2[2]) / 2)

    def create_graph(points):
        my_graph = nx.Graph()
        seen = {}
        count = 0
        for point_set in points:
            curr_nodes = []
            for point in point_set:
                if point not in seen.keys():
                    my_graph.add_node(count, pos = point)
                    seen[point] = count
                    curr_nodes.append(count)
                    count += 1
                else:
                    curr_nodes.append(seen[point])
            my_graph.add_edges_from([(curr_nodes[0], curr_nodes[1]),(curr_nodes[1], curr_nodes[2]),(curr_nodes[2], curr_nodes[0])])
        return my_graph

    def draw_triangle(graph):
        positions = nx.get_node_attributes(graph, 'pos')
        nodes = graph.nodes()
        to_remove = []
        nx.draw(graph, positions)
        pass
