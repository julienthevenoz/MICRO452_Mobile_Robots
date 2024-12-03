import numpy as np
from robot_api import Thymio
from tdmclient import ClientAsync
import math
class MotionControl:
  '''
  Control method: Proportional Control for Differential Drive Robots
  '''
  def __init__(self, thymio):
    # important parameter to adjust
    self.Ka = 50
    self.Kb = 0
    self.Kp = 20

    # speed limit
    self.max_velocity = 5000
    self.max_omega = 1000

    # units: mm
    self.wheel_radis = 20
    self.L = 95

    # robot interface
    self.thymio = thymio

    # parameters related to local navigation
    # self.threshold_high = 1000
    # self.threshold_low = 100

    # weights when implementing the local obstacle avoidance
    # self.obstSpeedGain = [0.3, 0.2, -0.1, -0.3, -0.4]

    # parameters related to global navigation
    self.goal_range = 30

  # def is_obstacle(self, prox_horizontal):
    
  #   mark = 0
  #   for i in range(5):
  #     if prox_horizontal[i] > self.threshold_high:
  #       return 1
  #     if prox_horizontal[i] < self.threshold_low:
  #       mark = mark + 1
  #   if mark == 5:
  #     return 0
  #   else:
  #     return 1

  # def obstacle_avoidance(self):
  #   speed = self.get_motor_speed()
  #   prox_horizontal = self.read_prox_sensors()
  #   while self.is_obstacle(prox_horizontal):
  #     delta = 0
  #     for i in range(5):
  #       delta += prox_horizontal[i] * self.obstSpeedGain[i]
  #     delta = max(-self.max_omega, min(delta, self.max_omega))
  #     speed[0] = speed[0] + delta
  #     speed[1] = speed[1] - delta
  #     speed[0] = int(speed[0])
  #     speed[1] = int(speed[1])
  #     self.thymio.set_motor_speed(speed[0], speed[1])
  #     speed = self.get_motor_speed()
  #     prox_horizontal = self.read_prox_sensors()

  # def move_to_next(self, current_pos, goal_pos, current_angle):
  #   '''
  #     caculate the speed based on current position and goal position
  #     Input: current_pos, goal_pos, current_angle
  #     Output: desired velocity and angular velocity of our robot
  #   '''
  #   delta = np.subtract(goal_pos, current_pos)
  #   dis = np.sqrt(np.sum(np.square(delta)))
  #   # need to check if this work
  #   theta = np.arctan2(delta[1], delta[0])
  #   angle_error = current_angle - theta
  #   v = self.Ka * dis
  #   omega = self.Kb * angle_error
  #   return v, omega

  def path_tracking(self, robot_state, goal_point):
    print('hi')
    if not robot_state or not goal_point:
      return False
    x, y, theta = robot_state
    
    theta = -theta
    print(theta)
    x_goal, y_goal = goal_point
    delta_x = x_goal - x
    delta_y = y_goal - y
    distance_to_goal = np.sqrt(delta_x ** 2 + delta_y ** 2)
    self.distance_to_goal = distance_to_goal
    
    # if distance_to_goal < self.goal_range:
    #   print('reached goal')
    #   return True
    angle_to_goal = np.arctan2(-delta_y, delta_x)
    
    alpha = - theta + angle_to_goal
    beta = - theta - alpha

    if(alpha > np.pi):
      alpha = alpha - 2 *np.pi
    elif (alpha < -np.pi):
      alpha = alpha + 2 * np.pi

    print(angle_to_goal)
    print(alpha)
    print(beta)
    # angle_error = angle_to_goal - theta
    # v = self.Ka * self.distance_to_goal * np.cos(math.degrees(angle_error))
    # omega = self.Kb * math.degrees(angle_error) + self.Ka * (np.cos(math.degrees(angle_error)) * np.sin(math.degrees(angle_error)))
    v = self.Kp * self.distance_to_goal
    omega = self.Ka * alpha + self.Kb * beta
    # v = max(-self.max_velocity, min(v, self.max_velocity))
    # omega = max(-self.max_omega, min(omega, self.max_omega))
    if(self.distance_to_goal > 500):
      v = 250 * self.wheel_radis
    if(self.distance_to_goal < 75):
      v = 100 * self.wheel_radis
    if(self.distance_to_goal < 10):
      v_L = int((self.L * omega / 2) / self.wheel_radis)
      v_R = int((self.L * omega / 2) / self.wheel_radis)
      self.set_motor_speed(v_L, v_R)
      return True
    v_L =  int((v - (self.L * omega / 2)) / self.wheel_radis)
    v_R = int((v + (self.L * omega / 2)) / self.wheel_radis)
    
    self.set_motor_speed(v_L, v_R)
    print(angle_to_goal,omega)
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
