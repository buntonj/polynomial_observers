import numpy as np


class PolyEstimator:
    '''
    Output estimator based on estimating a polynomial of degree d on the outputs.
    '''
    def __init__(self, d, N, dt):
        self.d = d
        self.N = N
        self.dt = dt
        self.regression_matrix = self.compute_matrix()

    def compute_matrix(self, t0=0.0):
        self.F = np.vander(np.linspace(t0, t0 + self.N*self.dt, self.N, endpoint=False), self.d, increasing=True)
        return np.linalg.pinv(self.F)

    def fit_global(self, y, t0):
        regression_matrix = self.compute_matrix(t0=t0)
        self.theta = regression_matrix @ y
        return self.theta

    def fit(self, y):
        '''
        input: y, an np.array of shape (N,) (i.e. scalar outputs)
        output: theta, degree-d polynomial coefficients
        '''
        self.theta = self.regression_matrix @ y
        return self.theta

    def estimate(self, t):
        '''
        input: t - a time to evaluate the current polyfit
        output: the current polynomial fit
        '''
        eval = 0.0
        for i in range(self.d):
            eval += self.theta[i]*t**float(i)
        return eval

    def differentiate(self, t, q):
        # q-th derivative of interpolant
        eval = 0.0
        for i in range(self.d-q):
            eval += self.theta[i+q]*(t**float(i))*np.prod(np.arange(i+q, i, -1))
        return eval


class TrajectoryEstimator:
    '''
    State estimator based on fitting simulation traces to output data.
    '''
    def __init__(self, trajectories: list):
        '''
        args:
            trajectories (list): list of tuples of numpy arrays, (x, y)
                x (np array, shape (n, num_samples))
                y (np array, shape (p, num_samples))
        '''
        self.num_trajectories = len(trajectories)
        self.num_samples = trajectories[0][0].shape[1]  # number of time steps in each simulation
        self.n = trajectories[0][0].shape[0]  # state dimension
        self.p = trajectories[0][1].shape[0]  # output dimension

        self.data_matrix = np.empty((self.num_samples, self.p*self.num_trajectories))
        self.prediction_matrix = np.empty((self.n, self.num_trajectories))

        for i, trajectory in enumerate(trajectories):
            self.data_matrix[:, i] = trajectory[1].flatten(order='C')
            self.prediction_matrix[:, i] = trajectory[0][:, -1]
        self.compute_matrix()

    def compute_matrix(self):
        self.regression_matrix = np.linalg.pinv(self.data_matrix)

    def fit(self, y):
        self.theta = self.regression_matrix @ y
        return self.theta

    def estimate(self):
        return self.prediction_matrix @ self.theta
