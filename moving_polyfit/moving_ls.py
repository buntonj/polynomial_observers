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
        F = np.vander(np.linspace(t0, t0 + self.N*self.dt, self.N, endpoint=False), self.d, increasing=True)
        return np.linalg.inv(F.T @ F) @ F.T

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
