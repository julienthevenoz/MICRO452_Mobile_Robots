import math
import time

class PIDController:

    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.prev_error = 0
        self.integral = 0
        self.index = 0

    def control(self, error):
        P = self.Kp * error
        self.integral = self.integral + error * self.dt
        I = self.Ki * self.integral
        D = self.Kd * ((error - self.prev_error) / self.dt)
        self.prev_error = error
        Gain = P + I + D

        return Gain
    
    def calculate_angle(self, path):
        dx = path[self.index + 1][0] - path[self.index][0]
        dy = path[self.index + 1][1] - path[self.index][1]
        angle =  math.atan2(dy, dx)
        angle_normalized = (angle + math.pi) % (2 * math.pi) - math.pi
        
        return angle_normalized
    
    def calculate_vel_control (self, angle_error, speed):
        PID = PIDController(10, 0, 0.1, 0.01)
        gain = PID.control(angle_error)
        vL = speed - gain
        vR = speed + gain

        if vL > 255:
            vL = 255
        
        if vR > 255:
            vR = 255
        
        if vL < 0:
            vL = 0
        
        if vR < 0:
            vL = 0

        return vL, vR
