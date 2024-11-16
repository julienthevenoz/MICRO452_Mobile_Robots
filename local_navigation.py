from tdmclient import ClientAsync, aw

class local_navigation:
  def __init__(self):
    self.threshold_high = None;
    self.threshold_low = None;
    
  def obstacle_avoidance(self, prox_horizontal, speed):
    '''
    Updating speed to avoid moving obatacles
    Possible methods: ANN weights / Potential fields
    '''
    return new_speed

  def is_obstacle_clear(self, prox_horizontal):
    '''
    Judging whether the obstacle is cleared or not
    If there is any value in prox_horizontal larger than threshold_high, return Flase
    If all values in prox_horizontal smaller than threshold_low, return True
    '''
    for sensor_value in prox_horizontal:
      if sensor_value > self.threshold_high:
        return False
      if sensor_value < self.threshold_low:
        
  def activate(self, prox_horizontal, speed):
    while is not self.is_obstacle_clear(prox_horizontal):
      speed = self.obstacle_avoidance(prox_horizontal, speed)
    break
      


