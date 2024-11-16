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
        
  def activate(self, prox_horizontal, speed):
    while is not self.is_obstacle_clear(prox_horizontal):
      speed = self.obstacle_avoidance(prox_horizontal, speed)
    break
      


