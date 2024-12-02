# Connect with the thymio
from tdmclient import ClientAsync
from global_navigation import GlobalNavigation

# client = ClientAsync()
# node = await client.wait_for_node()
# await node.lock()

from tdmclient import ClientAsync
from global_navigation import GlobalNavigation

# client = ClientAsync()
# node = await client.wait_for_node()
# await node.lock()

from Vision import Vision, show_many_img
import time
from motion_control import MotionControl
from robot_api import Thymio
from global_navigation import GlobalNavigation
import numpy as np
from kalman_filter import KalmanFilter


#create vision module object
visio = Vision()
visio.begin()

"""Point d'entrée principal"""
global_nav = GlobalNavigation()
# Thymio = Thymio(node, client)
# motion_control = MotionControl(Thymio)
kalman_filter =KalmanFilter()



thymio, goal, obstacles = [],[],[]

while not (thymio and goal and obstacles):
    _thymio, _goal, _obstacles = visio.get_thymio_goal_and_obstacles()
    time.sleep(0.2)
    if _thymio : 
        kal_state = _thymio
        thymio = _thymio
    if _goal :
        goal = _goal
    if _obstacles :
        obstacles = _obstacles

print("everything ok to start")

path, _, _ = global_nav.dijkstra(thymio, goal, obstacles)
# path.pop(0)
goal_point = path[1]
visio.camera_feed.vision_module.path = path

# try:
#     pre_variance = np.ones((3, 3))
#     pre_state = thymio
#     start_time = time.time()
#     while True:
#         _thymio, _goal, _obstacles = visio.get_thymio_goal_and_obstacles()
#         if _thymio : 
#             thymio = _thymio
#         if _goal :
#             goal = _goal
#         if _obstacles :
#             obstacles = _obstacles

#         motion_control.obstacle_avoidance()
#         if motion_control.path_tracking(kal_state, goal_point):
#            print("reached to goal")
#            path.pop(0)
#            if not path:
#                motion_control.stop_motor()
#                break
#            else:
#                goal_point = path[0]
#         thymio, goal, obstacles = visio.get_thymio_goal_and_obstacles()
#         if not thymio:
#             thymio = [0, 0, 0]
#             kalman_filter.Q =  np.array([[1000, 0, 0],
#                                          [0, 1000, 0],
#                                          [0, 0, 1000]])
#         motor_speed = motion_control.get_motor_speed()
#         end_time = time.time()
#         time_step = end_time - start_time
#         start_time = time.time()
#         vl_displacement = motor_speed[0] * 0.417 * time_step * 0.563 # 600/1065
#         vr_displacement = motor_speed[1] * 0.417 * time_step * 0.446 # 300/673
#         kal_state, kal_variance = kalman_filter.estimate(pre_state, pre_variance, vl_displacement, vr_displacement, thymio)
#         visio.camera_feed.past_positions.append(kal_state[:2])
#         pre_state = kal_state
#         pre_variance = kal_variance
#         time.sleep(0.2)
# except KeyboardInterrupt:
#     print("Stop the program")
# finally:
#     visio.stop()
#     print("Program finishes")