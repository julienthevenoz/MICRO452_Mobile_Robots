import numpy as np

class KalmanFilter:
    '''
    Kalman filter class
    the defined state of our robot[px, py, orientation]
    the defined input[vl, vr]
    '''
    def __init__(self):
        "Variance of the state measurement"
        self.Q = np.array([[0.1, 0, 0],
                           [0, 0.1, 0],
                           [0, 0, 0.1]])
        "Variance of the control input"
        self.R = np.array([[10, 0],
                           [0, 10]])

        "Distance between two wheels(m)"
        self.L = 0.095


    def get_A_linear(self, theta, v, omega):
        A_linear = np.array([
            [1, 0, -v *np.sin(theta + omega)],
            [0, 1, v * np.cos(theta + omega)],
            [0, 0, 1]
        ])
        return A_linear

    def get_B_linear(self, theta, v, omega):
        dx_dvl = 0.5 * np.cos(theta + omega) - v * np.sin(theta + omega) / (2 * self.L)
        dx_dvr = 0.5 * np.cos(theta + omega) + v * np.sin(theta + omega) / (2 * self.L)
        dy_dvl = 0.5 * np.sin(theta + omega) + v * np.cos(theta + omega) / (2 * self.L)
        dy_dvr = 0.5 * np.sin(theta + omega) - v * np.cos(theta + omega) / (2 * self.L)
        dtheta_dvl = 1 / (2 * self.L)
        dtheta_dvr = -1 / (2 * self.L)
        B_linear = np.array([
            [dx_dvl, dx_dvr],
            [dy_dvl, dy_dvr],
            [dtheta_dvl, dtheta_dvr]
        ])
        return B_linear

    def estimate(self, pre_state, pre_variance, vl, vr, measurenment):
        v = (vl + vr) / 2
        omega = (vl + vr) / (2 * self.L)
        "state update"
        est_state = np.array([pre_state[0] + v * np.sin(pre_state[2] + omega),
                              pre_state[1] + v * np.sin(pre_state[2] + omega),
                              (pre_state[2] + omega) % (2*np.pi)])
        A = self.get_A_linear(pre_state[2], v, omega)
        B = self.get_B_linear(pre_state[2], v, omega)
        est_variance = np.matmul(A, np.matmul(pre_variance, A.T)) + np.matmul(B, np.matmul(self.R, B.T))
        K = np.matmul(est_variance, np.linalg.inv(est_variance + self.Q))
        kal_state = est_state + np.matmul(K, (measurenment - est_state))
        kal_variance = est_variance - np.matmul(K, est_variance)
        return kal_state, kal_variance
    

    