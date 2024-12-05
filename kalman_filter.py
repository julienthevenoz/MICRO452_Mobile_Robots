import numpy as np

class KalmanFilter:
    '''
    Kalman filter class
    the defined state of our robot[px, py, orientation]
    the defined input[vl, vr]
    '''
    def __init__(self):
        # measurement noise
        self.R = np.array([[0.01, 0, 0],
                           [0, 0.01, 0],
                           [0, 0, 0.01]])

        self.R_crash = np.array([[np.inf, 0, 0],
                                 [0, np.inf, 0],
                                 [0, 0, np.inf]])

        self.R_normal = np.array([[0.01, 0, 0],
                           [0, 0.01, 0],
                           [0, 0, 0.01]])
        # state noise
        self.Q = np.array([[5, 0, 0],
                           [0, 5, 0],
                           [0, 0, 0.5]])

        "Dis‘tance between two wheels(mm)"
        self.L = 95

        self.A = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])

        self.H = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])

        self.kal_variance = np.array([[0.01, 0, 0],
                                      [0, 0.01, 0],
                                      [0, 0, 0.01]])

        self.kal_state = None
        self.S = None
        self.K = None


    def estimate(self, vl, vr, measurement):
        measurement = np.array(measurement)
        u = np.array([vl, vr])
        A = self.A
        B = np.array([[np.cos(self.kal_state[2]) / 2, np.cos(self.kal_state[2]) / 2],
                      [np.sin(self.kal_state[2]) / 2, np.sin(self.kal_state[2]) / 2],
                      [1 / (2 * self.L), - 1 / (2 * self.L) ]])
        # est_state = np.matmul(A, self.kal_state) + np.matmul(B, u)
        est_state = A @ (self.kal_state) + B @ u
        
        # est_variance = np.matmul(A, np.matmul(self.kal_variance , A.T)) + self.Q
        est_variance = A @ (self.kal_variance @ A.T) + self.Q
        # yk = measurement - np.matmul(self.H, est_state)
        yk = measurement - self.H @ est_state
        # self.S = np.matmul(self.H, np.matmul(est_variance, self.H.T)) + self.R
        self.S = self.H @ (est_variance @ self.H.T) + self.R
        # self.K = np.matmul(est_variance, np.matmul(self.H.T, np.linalg.inv(self.S)))
        self.K = est_variance @ (self.H.T @ np.linalg.inv(self.S))
        # self.kal_state = est_state + np.matmul(self.K, yk)
        self.kal_state = est_state + self.K @ yk
        # self.kal_variance = np.matmul(np.eye(3) - np.matmul(self.K, self.H), est_variance)
        self.kal_variance = (np.eye(3) - self.K @ self.H) @ est_variance
        return self.kal_state, self.kal_variance

    def check_camera(self, thymio):

        #if thymio is None or [], it means the camera isn't detecting anything 
        if not thymio:
            self.R = self.R_crash  #then we make R infinite so that the Kalman completely ignores the camera measurement
            return [-1, -1, -1]    #and we return an impossible value 
        #if not, keep R and thymio normal
        else:
            self.R = self.R_normal
            return thymio
