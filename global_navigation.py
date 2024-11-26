<<<<<<< HEAD
class GlobalNavigation:
  '''
  Could use some existing libraries: pyvisgraph
  Input: start_point, end_point, obstacles(get from visual module)
  Output: shortest path
  '''
  def __init__(self):

  def shortest_path(self,thymio, goal, obstacles):
    path = [0, 0, 0, 0,]
    return path
=======
import cv2
import numpy as np
from shapely.geometry import LineString, Polygon
from PIL import Image
from shapely import affinity
import matplotlib.pyplot as plt
import heapq
import geopandas as gpd
from geopandas import GeoSeries

class global_navigation():

    def __init__(self, start_point, goal_point, vertices_list):
            self.start_point = start_point
            self.goal_point = goal_point
            self.vertices_list = vertices_list
            self.margin = 20
            

    def createObstacle(self):
        # Create obstacles as polygons
        obstacle_polygons = [Polygon(vertices) for vertices in self.vertices_list]
        i = 0
        for obstacle in obstacle_polygons:
            print(obstacle[i])
            i = i + 1


        g = (GeoSeries(obstacle_polygons)).buffer(self.margin, join_style = 2)

        # Combine all vertices
        all_vertices = [self.start_point, self.goal_point] + [tuple(v) for vertices in self.vertices_list for v in vertices]
        m = 2

        for obstacle in g:
            j = 0
            while j < len(obstacle.exterior.coords) - 1:
                    all_vertices[m] = obstacle.exterior.coords[j]
                    j = j + 1
                    m = m + 1
        return obstacle_polygons, all_vertices


    # Function to check if an edge is valid, doesn't cross obstacles
    def is_edge_valid(p1, p2, obstacles):
        edge = LineString([p1, p2])
        for obstacle in obstacles:
            if edge.crosses(obstacle) or edge.within(obstacle):
                return False
        return True

    def visibilityGraph(self):
        obstacle_polygons, all_vertices = self.createObstacle()
        visibility_graph = {v: [] for v in all_vertices}
        for i, v1 in enumerate(all_vertices):
            for j, v2 in enumerate(all_vertices):
                if i != j and self.is_edge_valid(v1, v2, obstacle_polygons):
                    visibility_graph[v1].append(v2)
        return visibility_graph
                    

    # Implement Dijkstra's Algorithm
    def dijkstra(self):
        graph = self.visibilityGraph()
        queue = [(0, self.start_point)]  # (cost, current_node)
        distances = {node: float("inf") for node in graph}
        previous_nodes = {node: None for node in graph}
        distances[self.start_point] = 0

        while queue:
            current_distance, current_node = heapq.heappop(queue)

            if current_node == self.goal_point:
                path = []
                while current_node:
                    path.append(current_node)
                    current_node = previous_nodes[current_node]
                return path[::-1], current_distance

            for neighbor in graph[current_node]:
                edge_distance = np.linalg.norm(np.array(current_node) - np.array(neighbor))
                new_distance = current_distance + edge_distance

                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(queue, (new_distance, neighbor))

        return None, float("inf"), graph

    def plotGraph(self):

        shortest_path, path_distance, graph = self.dijkstra()


        plt.figure(figsize=(10, 10))
    
        plt.title("Visibility Graph and Shortest Path")

        for v1, neighbors in graph.items():
            for v2 in neighbors:
                plt.plot([v1[0], v2[0]], [v1[1], v2[1]], c="gray", linewidth=0.5)

        if shortest_path:
            for i in range(len(shortest_path) - 1):
                x1, y1 = shortest_path[i]
                x2, y2 = shortest_path[i + 1]
                plt.plot([x1, x2], [y1, y2], c="red", linewidth=2)

        plt.scatter(self.start_point[0], self.start_point[1], c="red", s=100, label="Start (Red)")
        plt.scatter(self.goal_point[0], self.goal_point[1], c="blue", s=100, label="Goal (Blue)")

        plt.legend()
        plt.axis("on")
        plt.show()

        print("Shortest Path:", shortest_path)
        print(shortest_path[1][1])
        print("Path Distance:", path_distance)





>>>>>>> cfe62d4d780d487184f5cecf24c9e73f5e63b604
