
class localNavigation():
  
  def __init__(self):
    
    self.threshold_high = 2000 
    self.threshold_low = 300

  def global_to_local (self, prox_horizontal, state):
    mark = 0
    if state == 0:
      for i in range(5):
        if prox_horizontal[i] > self.threshold_high:
          state = 1
    elif state == 1:
        for i in range(5):
          if prox_horizontal[i] < self.threshold_low:
            mark = mark + 1
        if mark == 5:
          state = 0
        else:
          state = 1
    return state
    
  def reactive_control(self, prox_horizontal, y):
    w_l = [40,  20, -20, -20, -40,  30, -10, 8, 0]
    w_r = [-40, -20, -20,  20,  40, -10, 30, 0, 8]

    sensor_scale = 900

    x = [0,0,0,0,0,0,0,0,0]
    
    x[7] = y[0] // 20
    x[8] = y[1] // 20

    for i in range(7):
            
            x[i] = prox_horizontal[i] // sensor_scale

            y[0] = y[0] + x[i] * w_l[i]
            y[1] = y[1] + x[i] * w_r[i]

    return y
      


