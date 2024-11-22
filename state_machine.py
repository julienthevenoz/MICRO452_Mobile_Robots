from win32gui import FlashWindowEx

from global_navigation import global_navigation
from robot_api import Thymio

class StateMachine:
  def __init__(self):
    self.state = 1
    self.previous_state = 0
    # parameters related to local navigation
    self.threshold_high = 1000
    self.threshold_low = 100
    
  def is_goal_reached(self):
    pass

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

  def show_state_info(self):
    if self.state != self.previous_state:
      if self.state == 2:
        print("Oops, there is an obstacle!")
      elif self.state == 1:
        print("The obstacle has cleared!")

  def state_update(self, prox_horizontal):
    self.previous_state = self.state
    if self.is_obstacle(prox_horizontal):
      self.state = 2
    else:
      self.state = 1
    self.show_state_info()

  # def run(self):
  #   while is not self.is_goal_reached():
  #     if self.state == 1:
  #       # glabal_navigation
  #       speed = self.global_nav.move_to_next(current_pos, goal_pos, current_angle)
  #       self.thymio.set_motor_speed(speed)
  #       self.state_update()
  #     if self.state == 2:
  #       # local navigation
  #       speed = self.local_navigation.obstacle_avoidance(self.thymio.prox_horizontal, self.thymio.speed)
  #       self.state_update()
  #     if self.state == 3:
  #       # reach the goal
  #       break
  


