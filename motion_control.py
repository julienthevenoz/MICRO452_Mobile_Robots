import numpy as np
from robot_api import Thymio
from tdmclient import ClientAsync
import time
class MotionControl:
  '''
  Control method: Proportional Control for Differential Drive Robots
  '''
  def __init__(self, thymio):
    # important parameter to adjust
    self.Ka = 30
    self.Kb = 40

    # speed limit
    self.max_velocity = 1000
    self.max_omega = 40

    # units: mm
    self.wheel_radis = 20
    self.L = 95

    # robot interface
    self.thymio = thymio

    # parameters related to local navigation
    self.threshold_high = 1000
    self.threshold_low = 100

    # weights when implementing the local obstacle avoidance
    self.obstSpeedGain = [0.3, 0.2, -0.1, -0.3, -0.4]

    # parameters related to global navigation
    self.goal_range = 10

    self.start_time = None
    self.end_time = None

  def is_obstacle(self, prox_horizontal):
    mark = 0
    for i in range(5):
      if prox_horizontal[i] > self.threshold_high:
        return True
      if prox_horizontal[i] < self.threshold_low:
        mark = mark + 1
    if mark == 5:
      return False
    else:
      return True

  def obstacle_avoidance(self):
    speed = self.get_motor_speed()
    prox_horizontal = self.read_prox_sensors()
    while self.is_obstacle(prox_horizontal):
      delta = 0
      for i in range(5):
        delta += prox_horizontal[i] * self.obstSpeedGain[i]
      delta = max(-self.max_omega, min(delta, self.max_omega))
      speed[0] = speed[0] + delta
      speed[1] = speed[1] - delta
      speed[0] = int(speed[0])
      speed[1] = int(speed[1])
      self.thymio.set_motor_speed(speed[0], speed[1])
      speed = self.get_motor_speed()
      prox_horizontal = self.read_prox_sensors()

  def path_tracking(self, robot_state, goal_point):
    if self.start_time is None:
      self.start_time = time.time()
    x, y, theta = robot_state
    x_goal, y_goal = goal_point
    delta_x = x_goal - x
    delta_y = y_goal - y
    distance_to_goal = np.sqrt(delta_x ** 2 + delta_y ** 2)
    self.distance_to_goal = distance_to_goal
    if distance_to_goal < self.goal_range:
      print('reached goal')
      return True
    angle_to_goal = np.arctan2(delta_y, delta_x)
    angle_error = angle_to_goal - theta
    # v = self.Ka * distance_to_goal
    v = 80
    omega = self.Kb * angle_error
    v = max(-self.max_velocity, min(v, self.max_velocity))
    omega = max(-self.max_omega, min(omega, self.max_omega))
    self.set_motor_speed(v+omega, v-omega)
    return False

  def get_displacement(self):
    motor_speed = self.get_motor_speed()
    self.end_time = time.time()
    time_step = self.end_time - self.start_time
    vl_displacement = motor_speed[0] * 0.417 * time_step * 0.563  # 600/1065
    vr_displacement = motor_speed[1] * 0.417 * time_step * 0.446  # 300/673
    self.start_time = time.time()
    return vl_displacement, vr_displacement

  def get_motor_speed(self):
    motor_speed = self.thymio.read_motors_speed()
    return motor_speed

  def set_motor_speed(self, left_speed, right_speed):
    left_speed = int(left_speed)
    right_speed = int(right_speed)
    self.thymio.set_motor_speed(left_speed, right_speed)

  def stop_motor(self):
    self.thymio.stop_robot()

  def read_prox_sensors(self):
    prox = self.thymio.read_prox_sensors()
    return prox
