class global_navigation:
  '''
  Could use some existing libraries: pyvisgraph
  Input: start_point, end_point, obstacles(get from visual module)
  Output: shortest path
  '''
import cv2
import numpy as np
from shapely.geometry import LineString, Polygon
from PIL import Image
from shapely import affinity
import matplotlib.pyplot as plt
import heapq
import geopandas as gpd
from geopandas import GeoSeries



image_path = r"C:\Users\User\OneDrive\Desktop\EPFL\MA1\Basics of Mobile Robotics\Project\Images\Paint_map.png"  # Update this path if needed
image_cv = cv2.imread(image_path)

gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

margin = 20


# Detect contours and approximate polygon vertices
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
vertices_list = []
for contour in contours:
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices_list.append(approx.reshape(-1, 2))


start_point = (200, 600)  
goal_point = (950, 550)  

# Create obstacles as polygons
obstacle_polygons = [Polygon(vertices) for vertices in vertices_list]

g = (GeoSeries(obstacle_polygons)).buffer(margin, join_style = 2)

# Combine all vertices
all_vertices = [start_point, goal_point] + [tuple(v) for vertices in vertices_list for v in vertices]

m = 2

for obstacle in g:
   j = 0
   while j < len(obstacle.exterior.coords) - 1:
        all_vertices[m] = obstacle.exterior.coords[j]
        j = j + 1
        m = m + 1


# Function to check if an edge is valid, doesn't cross obstacles
def is_edge_valid(p1, p2, obstacles):
    edge = LineString([p1, p2])
    for obstacle in obstacles:
        if edge.crosses(obstacle) or edge.within(obstacle):
            return False
    return True

# Construct the visibility graph
visibility_graph = {v: [] for v in all_vertices}
for i, v1 in enumerate(all_vertices):
    for j, v2 in enumerate(all_vertices):
        if i != j and is_edge_valid(v1, v2, obstacle_polygons):
            visibility_graph[v1].append(v2)

# Implement Dijkstra's Algorithm
def dijkstra(graph, start, goal):
    queue = [(0, start)]  # (cost, current_node)
    distances = {node: float("inf") for node in graph}
    previous_nodes = {node: None for node in graph}
    distances[start] = 0

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        if current_node == goal:
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

    return None, float("inf")


shortest_path, path_distance = dijkstra(visibility_graph, start_point, goal_point)


plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
plt.title("Visibility Graph and Shortest Path")


for vertices in vertices_list:
    for (x, y) in vertices:
        plt.scatter(x, y, c="green", s=20)  # Vertices

for x,y in all_vertices:
   
        plt.scatter(x, y, c="black", s=40)

for v1, neighbors in visibility_graph.items():
    for v2 in neighbors:
        plt.plot([v1[0], v2[0]], [v1[1], v2[1]], c="gray", linewidth=0.5)

if shortest_path:
    for i in range(len(shortest_path) - 1):
        x1, y1 = shortest_path[i]
        x2, y2 = shortest_path[i + 1]
        plt.plot([x1, x2], [y1, y2], c="red", linewidth=2)

plt.scatter(start_point[0], start_point[1], c="red", s=100, label="Start (Red)")
plt.scatter(goal_point[0], goal_point[1], c="blue", s=100, label="Goal (Blue)")

plt.legend()
plt.axis("on")
plt.show()

print("Shortest Path:", shortest_path)
print("Path Distance:", path_distance)

