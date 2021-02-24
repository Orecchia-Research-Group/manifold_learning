import numpy as np
import networkx as nx
import matplotlib.pyplot as pyplot

PHI = 1.61803

class icosahedron:
    def __init__(self):
        i_graph = nx.Graph()
        vertices = [(1, PHI, 0), (-1, PHI, 0), (1, -PHI, 0), (-1, -PHI, 0), (0, 1, PHI), (0, 1, -PHI), (0, -1, PHI), (0, -1, -PHI), (PHI, 0, 1), (-PHI, 0, 1), (PHI, 0, -1), (-PHI, 0, -1)]

        # Number of Sierpinski iterations desired
        num_iter = 1
        faces = []

        # Create triangular for icosahedron
        face_1 = triangle(num_iter, vertices[0], vertices[4], vertices[1])
        faces.append(face_1.create_graph())
        face_2 = triangle(num_iter, vertices[0], vertices[1], vertices[5])
        faces.append(face_2.create_graph())
        face_3 = triangle(num_iter, vertices[0], vertices[4], vertices[8])
        faces.append(face_3.create_graph())
        face_4 = triangle(num_iter, vertices[8], vertices[4], vertices[6])
        faces.append(face_4.create_graph())
        face_5 = triangle(num_iter, vertices[0], vertices[10], vertices[8])
        faces.append(face_5.create_graph())
        face_6 = triangle(num_iter, vertices[0], vertices[10], vertices[5])
        faces.append(face_6.create_graph())
        face_7 = triangle(num_iter, vertices[10], vertices[7], vertices[5])
        faces.append(face_7.create_graph())
        face_8 = triangle(num_iter, vertices[2], vertices[10], vertices[8])
        faces.append(face_8.create_graph())
        face_9 = triangle(num_iter, vertices[8], vertices[2], vertices[6])
        faces.append(face_9.create_graph())
        face_10 = triangle(num_iter, vertices[2], vertices[10], vertices[7])
        faces.append(face_10.create_graph())
        face_11 = triangle(num_iter, vertices[2], vertices[3], vertices[7])
        faces.append(face_11.create_graph())
        face_12 = triangle(num_iter, vertices[2], vertices[3], vertices[6])
        faces.append(face_12.create_graph())
        face_13 = triangle(num_iter, vertices[3], vertices[7], vertices[11])
        faces.append(face_13.create_graph())
        face_14 = triangle(num_iter, vertices[3], vertices[9], vertices[11])
        faces.append(face_14.create_graph())
        face_15 = triangle(num_iter, vertices[3], vertices[9], vertices[6])
        faces.append(face_15.create_graph())
        face_16 = triangle(num_iter, vertices[9], vertices[4], vertices[6])
        faces.append(face_16.create_graph())
        face_17 = triangle(num_iter, vertices[9], vertices[11], vertices[1])
        faces.append(face_17.create_graph())
        face_18 = triangle(num_iter, vertices[1], vertices[11], vertices[5])
        faces.append(face_18.create_graph())
        face_19 = triangle(num_iter, vertices[9], vertices[4], vertices[1])
        faces.append(face_19.create_graph())
        face_20 = triangle(num_iter, vertices[5], vertices[7], vertices[11])
        faces.append(face_20.create_graph())

        for face in faces:
            i_graph = nx.disjoint_union(i_graph, face)

        self.graph = i_graph
        


# Helper functions for triangle class
def triangulate(num_iterations, points):
        if num_iterations > 0:
            points.extend(triangulate(num_iterations - 1, [points[0], midpoint(points[0], points[1]), midpoint(points[0], points[2])]))
            points.extend(triangulate(num_iterations - 1, [midpoint(points[0], points[2]), midpoint(points[2], points[1]), points[2]]))
            points.extend(triangulate(num_iterations - 1, [midpoint(points[0], points[1]), points[1], midpoint(points[2], points[1])]))
            points.extend(triangulate(num_iterations - 1, [midpoint(points[0], points[2]), midpoint(points[2], points[1]), midpoint(points[1], points[0])]))
        return points

def midpoint(point1, point2):
    return ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2, (point1[2] + point2[2]) / 2)
        

# Defines triangular faces in 3-dimensions
class triangle:
    def __init__(self, num_iterations, node1 = (0.0, 0.0, 0.0), node2 = (2.0, 0.0, 0.0), node3 = (1.0, 1.0, 0.0)):
        output_points = triangulate(num_iterations, [node1, node2, node3])
        final_points = []
        for count, point in enumerate(output_points):
            if count % 3 == 0:
                final_points.append(output_points[count:count + 3])
        
        self.points = final_points

    def create_graph(self):
        points = self.points
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
        self.graph = my_graph
        pass

    # Note: nx.draw does not work in 3-dimensions, projection to 2-dimensions required
    def draw_triangle(self):
        graph = self.graph
        positions = nx.get_node_attributes(graph, 'pos')
        nodes = graph.nodes()
        to_remove = []
        nx.draw(graph, positions)
        pass
