import numpy as np
class LocalNavigation:
  def __init__(self):
    self.obstSpeedGain = [0.6, 0.4, -0.2, -0.6, -0.8]

  def obstacle_avoidance(self, prox_horizontal, speed, current_pos, goal_pos):
    '''
    Updating speed to avoid moving obatacles
    Local navigation method: Potential fields(taken from exercise session 4)
    Input: prox_horizontal and speed
    Output: new speed of our robot(must be int)
    '''
    new_speed = speed
    # repulsive force from obstacles
    for i in range(6):
      new_speed[0] = new_speed[0] + prox_horizontal[i] * self.obstSpeedGain[i]
      new_speed[1] = new_speed[1] + prox_horizontal[i] * self.obstSpeedGain[i]
    # attractive force from goal
    dis_x = goal_pos[0] - current_pos[0]
    dis_y = goal_pos[1] - current_pos[1]
    angle_to_goal = np.arctan2(-dis_y, dis_x)
    new_speed[0] = new_speed[0] + angle_to_goal
    new_speed[1] = new_speed[1] - angle_to_goal
    new_speed[0] = int(new_speed[0])
    new_speed[1] = int(new_speed[1])
    return new_speed

  def obstacle_avoidance_test(self, prox_horizontal):
    new_speed = np.array([30, 30])
    for i in range(5):
      new_speed[0] = new_speed[0] + prox_horizontal[i] * self.obstSpeedGain[i]
      new_speed[1] = new_speed[1] - prox_horizontal[i] * self.obstSpeedGain[i]
    new_speed[0] = int(new_speed[0])
    new_speed[1] = int(new_speed[1])
    return new_speed


