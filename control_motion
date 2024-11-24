import numpy as np
from robot_api import Thymio
from tdmclient import ClientAsync
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
    self.obstSpeedGain = [6, 4, -2, -6, -8]

    # parameters related to global navigation
    self.goal_range = 2.0

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
    if self.is_obstacle(prox_horizontal):
      for i in range(5):
        speed[0] = speed[0] + prox_horizontal[i] * self.obstSpeedGain[i]
        speed[1] = speed[1] - prox_horizontal[i] * self.obstSpeedGain[i]
      speed[0] = int(speed[0])
      speed[1] = int(speed[1])
      self.thymio.set_motor_speed(speed[0], speed[1])
    else:
      self.thymio.set_motor_speed(30, 30)


  def move_to_next(self, current_pos, goal_pos, current_angle):
    '''
      caculate the speed based on current position and goal position
      Input: current_pos, goal_pos, current_angle
      Output: desired velocity and angular velocity of our robot
    '''
    delta = np.subtract(goal_pos, current_pos)
    dis = np.sqrt(np.sum(np.square(delta)))
    # need to check if this work
    theta = np.arctan2(delta[1], delta[0])
    angle_error = current_angle - theta
    v = self.Ka * dis
    omega = self.Kb * angle_error
    return v, omega

  def path_tracking(self, robot_state, goal_point):
    x, y, theta = robot_state
    x_goal, y_goal = goal_point
    delta_x = x_goal - x
    delta_y = y_goal - y
    distance_to_goal = np.sqrt(delta_x ** 2 + delta_y ** 2)
    if distance_to_goal < self.goal_range:
      print('reached goal')
      return True
    angle_to_goal = np.arctan2(delta_y, delta_x)
    angle_error = angle_to_goal - theta
    v = self.Ka * distance_to_goal
    omega = self.Kb * angle_error
    v = max(-self.max_velocity, min(v, self.max_velocity))
    omega = max(-self.max_omega, min(omega, self.max_omega))
    self.set_motor_speed(v+omega, v-omega)
    return False


  def get_motor_speed(self):
    motor_speed = self.thymio.read_motors_speed()
    return motor_speed

  def set_motor_speed(self, left_speed, right_speed):
    self.thymio.set_motor_speed(left_speed, right_speed)

  def stop_motor(self):
    self.thymio.stop_robot()

  def read_prox_sensors(self):
    prox = self.thymio.read_prox_sensors()
    return prox
