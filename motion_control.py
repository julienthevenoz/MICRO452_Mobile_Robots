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
    self.Ka = 50
    self.Kb = 0
    self.Kp = 20

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
    self.goal_range = 30

    self.start_time = None
    self.end_time = None

    self.max_velocity = 1000
    self.max_omega = 40

  def is_obstacle(self):
    mark = 0
    prox_horizontal = self.read_prox_sensors()
    for i in range(5):
      if prox_horizontal[i] > self.threshold_high:
        return 0
      if prox_horizontal[i] < self.threshold_low:
        mark = mark + 1
    if mark == 5:
      return 1
    else:
      return 0

  def obstacle_avoidance(self):
    speed = self.get_motor_speed()
    prox_horizontal = self.read_prox_sensors()
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
    
    theta = -theta
    x_goal, y_goal = goal_point
    delta_x = x_goal - x
    delta_y = y_goal - y
    distance_to_goal = np.sqrt(delta_x ** 2 + delta_y ** 2)
    self.distance_to_goal = distance_to_goal
    
    angle_to_goal = np.arctan2(-delta_y, delta_x)
    
    alpha = - theta + angle_to_goal
    beta = - (theta + alpha)

    if(alpha > np.pi):
      alpha = alpha - 2 *np.pi
    elif (alpha < -np.pi):
      alpha = alpha + 2 * np.pi

    v = self.Kp * self.distance_to_goal
    omega = self.Ka * alpha + self.Kb * beta

    if(self.distance_to_goal > 500):
      v = 250 * self.wheel_radis
    if(self.distance_to_goal < 50):
      v = 100 * self.wheel_radis
    if(self.distance_to_goal < self.goal_range):
      v_L = int((self.L * omega / 2) / self.wheel_radis)
      v_R = int((self.L * omega / 2) / self.wheel_radis)

      self.set_motor_speed(v_L, v_R)
      return True
    
    v_L =  int((v - (self.L * omega / 2)) / self.wheel_radis)
    v_R = int((v + (self.L * omega / 2)) / self.wheel_radis)

    self.set_motor_speed(v_L, v_R)
    return False



  def get_displacement(self):
    motor_speed = self.get_motor_speed()
    self.end_time = time.time()
    time_step = self.end_time - self.start_time
    vl_displacement = motor_speed[0] * 0.417 * time_step * 0.563  # 600/1065
    vr_displacement = motor_speed[1] * 0.417 * time_step * 0.446  # 300/673
    self.start_time = time.time()
    return vl_displacement, vr_displacement

  def is_kidnapping(self, kal_state, pre_state):
    x, y, theta = kal_state
    x_pre, y_pre, theta_pre = pre_state
    delta_x = x - x_pre
    delta_y = y - y_pre
    distance = np.sqrt(delta_x ** 2 + delta_y ** 2)
    if distance > 50:
      return True
    return False

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