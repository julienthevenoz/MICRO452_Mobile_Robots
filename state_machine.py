class state_machine:
  def __init__():
    self.thymio = MyThymio;
    self.state = state;
    
    # parameters related to local navigation
    self.threshold_high = None;
    self.threshold_low = None;
    
  def is_goal_reached(self):
    
  def is_obstacle_clear(self):

  def state_update(self):

  def run(self):
    while is not self.is_goal_reached():
      if state == 1:
        # glabal_navigation
        self.state_update()
      if state == 2:
        # local navigation
        speed = self.local_navigation.obstacle_avoidance(self.thymio.prox_horizontal, self.thymio.speed)
        self.state_update()
      if state == 3:
        # reach the goal 
        break
  

