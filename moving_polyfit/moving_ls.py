import numpy as np
from numpy.polynomial import Polynomial as P


class PolyEstimator:
    '''
    Output estimator based on estimating a polynomial of degree d on the outputs.
    '''
    def __init__(self, d, N, dt):
        self.d = d
        self.N = N
        self.dt = dt

    def fit(self, y, t0=0.0):
        '''
        Convenience fitting function wrapper.
        If t0 is not provided, does the fitting in ``local coordinates''
        i.e., treats the first index as a sample from t0=0.0.
        '''
        self.polynomial = P.fit(np.linspace(t0, t0+self.N*self.dt, self.N, endpoint=False), y, self.d)
        return self.polynomial.coef

    def estimate(self, t):
        '''
        input: t - a time to evaluate the current polyfit
        output: the current polynomial fit
        '''
        eval = self.polynomial(t)
        return eval

    def differentiate(self, t, q):
        # q-th derivative of polynomial fit
        eval = self.polynomial.deriv(q)(t)
        return eval


class TrajectoryEstimator:
    '''
    State estimator based on fitting simulation traces to output data.
    '''
    def __init__(self, trajectories: list, dt: float):
        '''
        args:
            trajectories (list): list of tuples of numpy arrays, (x, y)
                x (np array, shape (n, num_steps))
                y (np array, shape (p, num_steps))
        '''
        self.num_trajectories = len(trajectories)
        self.num_steps = trajectories[0][0].shape[1]  # number of time steps in each simulation
        self.n = trajectories[0][0].shape[0]  # state dimension
        self.p = trajectories[0][1].shape[0]  # output dimension

        self.A_matrix = np.zeros((self.n, self.n))
        for i in range(self.n-1):
            self.A_matrix[i, i+1] = float(self.num_steps)*dt

        self.output_data_matrix = np.empty((self.num_steps*self.p, self.num_trajectories))
        self.state_data_matrix = np.empty((self.num_steps*self.n, self.num_trajectories))
        self.prediction_matrix_end = np.empty((self.n, self.num_trajectories))
        self.prediction_matrix_start = np.empty((self.n, self.num_trajectories))

        for i, trajectory in enumerate(trajectories):
            self.output_data_matrix[:, i] = trajectory[1].flatten(order='F')
            self.state_data_matrix[:, i] = trajectory[0].flatten(order='F')
            self.prediction_matrix_end[:, i] = trajectory[0][:, -1]
            self.prediction_matrix_start[:, i] = trajectory[0][:, 0]
        self.compute_output_matrix()
        self.compute_state_matrix()

    def compute_state_matrix(self):
        self.state_regression_matrix = np.linalg.pinv(self.state_data_matrix)

    def compute_output_matrix(self):
        self.regression_matrix = np.linalg.pinv(self.output_data_matrix)

    def fit(self, y):
        self.theta = np.linalg.lstsq(self.output_data_matrix, y.flatten(order='F'), rcond=None)[0]
        return self.theta

    def fit_state(self, x):
        self.theta_state = np.linalg.lstsq(self.state_data_matrix, x.flatten(order='F'), rcond=None)[0]
        return self.theta_state

    def estimate_state_thetas(self):
        return self.prediction_matrix_end @ self.theta_state

    def estimate(self):
        return self.prediction_matrix_end @ self.theta
