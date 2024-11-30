from tdmclient import ClientAsync

from global_navigation import GlobalNavigation

client = ClientAsync()
node = await client.wait_for_node()
await node.lock()

from Vision import Vision, show_many_img
import time
from motion_control import MotionControl
from robot_api import Thymio
from global_navigation import GlobalNavigation
import numpy as np

#create vision module object
visio = Vision()
visio.begin()

"""Point d'entrée principal"""
global_nav = GlobalNavigation()
#Thymio = Thymio(node, client)
motion_control = MotionControl(Thymio)

#path.pop(0)
#goal_point = path[1]
#print(goal_point)


thymio, goal, obstacles = [],[],[]

while not (thymio and goal and obstacles):
    _thymio, _goal, _obstacles = visio.get_thymio_goal_and_obstacles()
    time.sleep(0.2)
    if _thymio : 
        thymio = _thymio
    if _goal :
        goal = _goal
    if _obstacles :
        obstacles = _obstacles

print("everything ok to start")

try:
    while True:
        _thymio, _goal, _obstacles = visio.get_thymio_goal_and_obstacles()
        if _thymio : 
            thymio = _thymio
        if _goal :
            goal = _goal
        if _obstacles :
            obstacles = _obstacles

        path, _, _ = global_nav.dijkstra(thymio, goal, obstacles)
        visio.analysis.path = path
        goal_point = path[1]
        #motion_control.obstacle_avoidance()
        #if motion_control.path_tracking(thymio, goal_point):
        #    if not path:
        #        break
        #    else:
        #        goal_point = path[0]
        # thymio, goal, obstacles = visio.get_thymio_goal_and_obstacles()
        time.sleep(0.2)
except KeyboardInterrupt:
    print("Stop the program")
finally:
    visio.stop()
    print("Program finishes")