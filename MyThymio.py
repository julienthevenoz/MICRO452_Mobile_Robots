#import Kalman_filter   #for example when we finish the filter we import it here
#import Local_Navigator

class MyThymio():
    def __init__(self):
        self.position = None
        self.map = None
        self.filter_module = None #Kalman_Filter()                  #then we give the filter object as an attribute of our thymio
        self.local_navigation_module = None  #Local_Navigator()     


    def set_speed(self,speed):
        '''Sets the speed of the wheels'''
        # set motor1 to speed[0] and motor2 to speed[2]
        pass


    def get_sensor_reading(self,sensor_name):
        ''' Returns a list which contains the reading of the desired sensor'''
        reading = []
        return reading


    def fuse_measurements(self,measurement1,measurement2):    #then we have a function of the thymio which calls the appropriate function of the Kalman_filter
        '''This function takes odometry measure and position measure from the camera.
        It returns an updated measurement of the position, estimated by the Kalman filter'''
        #self.filter_module.filter_some_shit(measurement1,measurement2)
        estimated_position = None
        return estimated_position
    

    def avoid_obstacle(self):
        '''Starts the local navigation obstacle avoidance mode based on potential fields (or whatever)'''
        #self.local_navigation_module.activate()
        pass
