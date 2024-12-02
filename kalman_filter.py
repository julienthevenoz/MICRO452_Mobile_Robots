import numpy as np

class KalmanFilter:
    '''
    Kalman filter class
    the defined state of our robot[px, py, orientation]
    the defined input[vl, vr]
    '''
    def __init__(self):
        "Variance of the state measurement"
        self.Q = np.array([[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]])
        "Variance of the control input"
        self.R = np.array([[1000, 0],
                           [0, 1000]])

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
        est_state = np.array([pre_state[0] + v * np.cos(pre_state[2] + omega),
                              pre_state[1] + v * np.sin(pre_state[2] + omega),
                              (pre_state[2] + omega) % (2*np.pi)])
        A = self.get_A_linear(pre_state[2], v, omega)
        B = self.get_B_linear(pre_state[2], v, omega)
        est_variance = np.matmul(A, np.matmul(pre_variance, A.T)) + np.matmul(B, np.matmul(self.R, B.T))
        K = np.matmul(est_variance, np.linalg.inv(est_variance + self.Q))
        kal_state = est_state + np.matmul(K, (measurenment - est_state))
        kal_variance = np.matmul(np.eye(3) - K, est_variance)
        return kal_state, kal_variance

def test_kalman_filter():
        # Initialize the Kalman Filter
        kf = KalmanFilter()

        # Time steps and simulation parameters
        time_steps = 100
        dt = 0.1  # Time step duration

        # Initialize state, variance, and lists for visualization
        true_state = np.array([0.0, 0.0, 0.0])  # [px, py, orientation]
        pre_state = true_state.copy()
        pre_variance = np.eye(3) * 0.1
        estimated_states = []
        true_states = []
        noisy_measurements = []

        # Control inputs (constant velocities)
        # The actual state of oy robot
        vl = 80  # Left wheel velocity
        vr = 80  # Right wheel velocity
        theta = 0
        v = (vl + vr) / 2
        omega = (vl - vr) / kf.L



        # Simulate motion and filtering
        for _ in range(time_steps):
            # Simulate true motion
            v = (vl + vr) / 2
            omega = (vr - vl) / (2 * kf.L)
            true_state = np.array([
                true_state[0] + v * np.cos(omega) * dt,
                true_state[1] + v * np.sin(omega) * dt,
                (true_state[2] + omega * dt) % (2 * np.pi)
            ])

            # Simulate noisy measurement
            measurement_noise = np.random.multivariate_normal(mean=[0, 0, 0], cov=kf.Q)
            noisy_measurement = true_state + measurement_noise

            control_noise = np.random.multivariate_normal(mean=[0, 0], cov=kf.R)
            vl_get = vl + control_noise[0]
            vr_get = vr + control_noise[1]

            vl_dis = vl_get * dt
            vr_dis = vr_get * dt

            # Kalman Filter estimation
            kal_state, kal_variance = kf.estimate(pre_state, pre_variance, vl_dis, vr_dis, noisy_measurement)

            # Store results
            true_states.append(true_state)
            noisy_measurements.append(noisy_measurement)
            estimated_states.append(kal_state)

            # Update the previous state and variance for the next step
            pre_state = kal_state
            pre_variance = kal_variance

        # Convert lists to numpy arrays for easier plotting
        true_states = np.array(true_states)
        noisy_measurements = np.array(noisy_measurements)
        estimated_states = np.array(estimated_states)

        # Plot results
        plt.figure(figsize=(10, 6))

        # Plot X vs Y (trajectory)
        plt.subplot(2, 1, 1)
        plt.plot(true_states[:, 0], true_states[:, 1], label="True Trajectory", color="green")
        plt.scatter(noisy_measurements[:, 0], noisy_measurements[:, 1], label="Noisy Measurements", color="red", s=10,
                    alpha=0.5)
        plt.plot(estimated_states[:, 0], estimated_states[:, 1], label="Estimated Trajectory", color="blue")
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.legend()
        plt.title("Kalman Filter: Trajectory")

        # Plot Orientation
        plt.subplot(2, 1, 2)
        plt.plot(range(time_steps), true_states[:, 2], label="True Orientation", color="green")
        plt.plot(range(time_steps), noisy_measurements[:, 2], label="Noisy Measurements", color="red", alpha=0.5)
        plt.plot(range(time_steps), estimated_states[:, 2], label="Estimated Orientation", color="blue")
        plt.xlabel("Time Step")
        plt.ylabel("Orientation (rad)")
        plt.legend()
        plt.title("Kalman Filter: Orientation")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Run the test function
    test_kalman_filter()
