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

        self.R_crash = np.array([[1000, 0, 0],
                                 [0, 1000, 0],
                                 [0, 0, 1000]])

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

        self.kal_state = None
        self.kal_variance = None


    def estimate(self, vl, vr, measurement):
        measurement = np.array(measurement).reshape(3, 1)
        u = np.array([[vl], [vr]])
        A = self.A
        B = np.array([[np.cos(self.kal_state[2]) / 2, np.cos(self.kal_state[2]) / 2],
                      [np.sin(self.kal_state[2]) / 2, np.sin(self.kal_state[2]) / 2],
                      [1 / (2 * self.L), - 1 / (2 * self.L) ]])
        est_state = np.matmul(A, self.kal_state) + np.matmul(B, u)
        est_variance = np.matmul(A, np.matmul(self.kal_variance , A.T)) + self.Q
        yk = measurement - np.matmul(self.H, est_state)
        S = np.matmul(self.H, np.matmul(est_variance, self.H.T)) + self.R
        K = np.matmul(est_variance, np.matmul(self.H.T, np.linalg.inv(S)))
        self.kal_state = est_state + np.matmul(K, yk)
        self.kal_variance = np.matmul(np.eye(3) - np.matmul(K, self.H), est_variance)
        return self.kal_state, self.kal_variance

    def check_camera(self, thymio):
        if not thymio:
            self.R = self.R_crash
            return [0, 0, 0]
        else:
            self.R = self.R_normal
            return thymio



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
            kal_state, kal_variance = kf.estimate(vl_dis, vr_dis, noisy_measurement)

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
