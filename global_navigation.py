import cv2
import numpy as np
from shapely.geometry import LineString, Polygon
from PIL import Image
from shapely import affinity
import matplotlib.pyplot as plt
import heapq
import geopandas as gpd
from geopandas import GeoSeries

class GlobalNavigation:

 def __init__(self):
  self.margin = 0
            

 def getObstacles(self, start, goal, obstacles):
  obstacle_polygons = [Polygon(vertices) for vertices in obstacles]
  buffered_obstacles = [obstacle.buffer(self.margin, join_style = "mitre") for obstacle in obstacle_polygons]
  all_vertices = [tuple(start), tuple(goal)]
  for buffered_obstacle in buffered_obstacles:
            for coord in buffered_obstacle.exterior.coords:
                all_vertices.append(tuple(coord))
  return obstacle_polygons, all_vertices


 # Function to check if an edge is valid, doesn't cross obstacles
 def is_edge_valid(self, p1, p2, obstacles):
  edge = LineString([p1, p2])
  for obstacle in obstacles:
   if edge.crosses(obstacle) or edge.within(obstacle):
    return False
  return True

 def visibilityGraph(self, start, goal, obtacles):
  obstacle_polygons, all_vertices = self.getObstacles(start, goal, obtacles)
  visibility_graph = {v: [] for v in all_vertices}
  for i, v1 in enumerate(all_vertices):
   for j, v2 in enumerate(all_vertices):
    if i != j and self.is_edge_valid(v1, v2, obstacle_polygons):
     visibility_graph[v1].append(v2)
  return visibility_graph


# Implement Dijkstra's Algorithm
 def dijkstra(self, thymio, goal, obstacles):

    thymio_x, thymio_y, theta = thymio
    start = (thymio_x, thymio_y)
    goal = tuple(goal)
    obstacles = [tuple(obstacle) for obstacle in obstacles]

    graph = self.visibilityGraph(start, goal, obstacles)
    queue = [(0, start)]  # (cost, current_node)
    distances = {node: float("inf") for node in graph}
    previous_nodes = {node: None for node in graph}
    distances[start] = 0

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        # If we've reached the goal, reconstruct the path
        if current_node == goal:
            path = []
            while current_node:
                path.append(current_node)
                current_node = previous_nodes[current_node]
            return path[::-1], current_distance, graph  # Return reversed path and distance

        # Check all neighbors
        for neighbor in graph[current_node]:
            edge_distance = np.linalg.norm(np.array(current_node) - np.array(neighbor))
            new_distance = current_distance + edge_distance

            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(queue, (new_distance, neighbor))

    return None, float("inf"), graph  # Return None if no path is found

 def plot_path(self, path, obstacles):

     # Visualize the result
     fig, ax = plt.subplots()

     # Plot obstacles
     for obstacle in obstacles:
         poly = Polygon(obstacle)
         x, y = poly.exterior.xy
         ax.fill(x, y, color='gray', alpha=0.5)

     # # Plot the start and goal points
     # ax.plot(start[0], start[1], 'go', label="Start")
     # ax.plot(goal[0], goal[1], 'ro', label="Goal")

     # Plot the path if one is found
     if path:
         path_x = [point[0] for point in path]
         path_y = [point[1] for point in path]
         ax.plot(path_x, path_y, 'b-', label="Path")

     ax.set_xlabel('X')
     ax.set_ylabel('Y')
     ax.set_title('Dijkstra Path Planning')
     ax.legend()

     plt.show()



# Test function for the GlobalNavigation class
if __name__ == "__main__":
    # Define some obstacles
    # obstacles = [
    #     [(2, 2), (4, 2), (4, 4), (2, 4)],  # Square obstacle
    #     [(6, 6), (8, 6), (8, 8), (6, 8)]  # Another square obstacle
    # ]
    # obstacles = []
    #
    # # Start and goal points
    # start = (0, 0)
    # goal = (10, 10)
    thymio = [73.973694, 75.61618, 1.055]
    goal = [406.53833, 69.10376]
    obstacles = [[[464, 321], [366, 321], [404, 238]]]

    # Create an instance of the GlobalNavigation class
    navigation = GlobalNavigation()

    # Run Dijkstra's algorithm to find the shortest path
    path, cost, graph = navigation.dijkstra(thymio, goal, obstacles)

    # Print results
    if path:
        print("Path found:", path)
        print("Total cost:", cost)
    else:
        print("No path found")
    navigation.plot_path(path, obstacles)