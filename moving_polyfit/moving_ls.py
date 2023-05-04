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
        self.residuals = np.empty((self.N,))

    def fit(self, y, t0=0.0):
        '''
        Convenience fitting function wrapper.
        If t0 is not provided, does the fitting in ``local coordinates''
        i.e., treats the first index as a sample from t0=0.0.
        '''
        t = np.linspace(t0, t0+self.N*self.dt, self.N, endpoint=False)
        self.polynomial = P.fit(t, y, self.d)

        # compute and save the fit residuals
        for i in range(y.shape[0]):
            self.residuals[i] = np.abs(self.polynomial(t[i]) - y[i])

        return self.polynomial.coef

    def estimate(self, t):
        '''
        input: t - a time to evaluate the current polyfit
        output: the current polynomial fit
        '''
        eval = self.polynomial(t)
        return eval

    def differentiate(self, t: float, q: int):
        # q-th derivative of polynomial fit
        eval = self.polynomial.deriv(q)(t)
        return eval


class TrajectoryEstimator:
    '''
    State estimator based on fitting simulation traces to output data.
    '''
    def __init__(self, y: np.ndarray, dt: float):
        '''
        args:
            y (np array, (p, d, num_steps, N) : N sims of p-dim output (scalar here) and its d derivatives for num_steps
            dt (float) : time step between samples in trajectories
        '''
        self.p = y.shape[0]  # output dimension, fixed at scalar for now
        self.d = y.shape[1]  # how many derivatives can we predict?
        self.num_steps = y.shape[2]  # number of time steps in each simulation
        self.num_trajectories = y.shape[3]  # number of simulations we are using

        self.dt = dt
        self.time = np.linspace(0.0, self.num_steps*dt, num=self.num_steps, endpoint=False)

        # TODO: accomodate non-scalar outputs with flattening
        self.estimation_matrix = np.empty((self.p*self.d, self.num_trajectories))

        self.build_fitting_matrix(y)
        self.build_estimation_matrix(y)

    def build_fitting_matrix(self, y):
        self.fitting_matrix = np.empty((self.num_steps*self.p, self.num_trajectories))
        for i in range(self.num_trajectories):
            self.fitting_matrix[:, i] = y[:, 0, :, i].flatten(order='F')

    def build_estimation_matrix(self, y):
        self.estimation_matrix = np.empty((self.p*self.d, self.num_steps, self.num_trajectories))
        for i in range(self.num_trajectories):
            self.estimation_matrix[:, :, i] = y[:, :, :, i].reshape((self.p*self.d, self.num_steps), order='F')

    def fit(self, y):
        self.theta = np.linalg.lstsq(self.fitting_matrix, y.flatten(order='F'), rcond=None)[0]
        self.residuals = np.abs(self.fitting_matrix @ self.theta - y.flatten(order='F'))
        return self.theta

    def estimate(self, t: float):
        return self.differentiate(t, 0)

    def nearest_times(self, t: float, k: int):
        idx = np.argpartition(np.abs(self.time - t), k)
        return self.time[idx[:k]], idx[:k]

    def differentiate(self, t: float, q: int):
        # extract the nearest times, linearly interpolate them
        # this math is valid when t is inside the interval of simulation evaluations
        # TODO: fix math for linear interp outside of interval
        # could possibly do fancier interpolations, with cubic splines, etc.
        times, ii = self.nearest_times(t, 2)
        w1 = 1.0 - np.abs(times[0] - t)/self.dt
        w2 = 1.0 - w1

        return (w1*self.estimation_matrix[q, ii[0], :] + w2*self.estimation_matrix[q, ii[1], :]) @ self.theta
