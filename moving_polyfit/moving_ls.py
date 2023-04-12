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
    def __init__(self, trajectories: list):
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

        self.data_matrix = np.empty((self.num_steps*self.p, self.num_trajectories))
        self.state_data_matrix = np.empty((self.num_steps*self.n, self.num_trajectories))
        self.prediction_matrix = np.empty((self.n, self.num_trajectories))

        for i, trajectory in enumerate(trajectories):
            self.data_matrix[:, i] = trajectory[1].flatten(order='F')
            self.state_data_matrix[:, i] = trajectory[0].flatten(order='F')
            self.prediction_matrix[:, i] = trajectory[0][:, -1]
        self.compute_matrix()
        self.compute_state_matrix()

    def compute_state_matrix(self):
        self.state_regression_matrix = np.linalg.pinv(self.state_data_matrix)

    def compute_matrix(self):
        self.regression_matrix = np.linalg.pinv(self.data_matrix)

    def fit(self, y):
        self.theta = self.regression_matrix @ y
        return self.theta

    def fit_state(self, x):
        self.theta_state = self.state_regression_matrix @ x.flatten(order='F')
        return self.theta_state

    def estimate(self):
        return self.prediction_matrix @ self.theta
