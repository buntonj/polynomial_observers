import numpy as np


class ControlAffineODE:
    def __init__(self, state_dim: int, input_dim: int, output_dim=1, f=None, g=None, h=None):
        self.n = state_dim
        self.m = input_dim
        self.p = output_dim

        # if no drift function is supplied, then set it to zero
        if f is None:
            if g is None:
                self.rhs = self.zero_rhs
            else:
                self.g = g
                self.rhs = self.zero_f_rhs
        else:
            self.f = f
            if g is None:
                self.rhs = self.zero_g_rhs
            else:
                self.g = g
                self.rhs = self.std_rhs

        # if no output is provided, then set it to zero
        if h is None:
            self.h = self.no_output
        else:
            self.h = h

    def std_rhs(self, t: float, x: np.ndarray, u: np.ndarray):
        return self.f(t, x) + self.g(t, x)@u

    def zero_f_rhs(self, t: float, x: np.ndarray, u: np.ndarray):
        return self.g(t, x)@u

    def zero_g_rhs(self, t: float, x: np.ndarray, u: np.ndarray):
        return self.f(t, x)

    def zero_rhs(self, t: float, x: np.ndarray, u: np.ndarray):
        return np.zeros((self.n,))

    def no_output(self, t: float, x: np.ndarray, u: np.ndarray):
        return 0.0


class Integrator(ControlAffineODE):
    def __init__(self, state_dim: int, output_fn=None, output_derivative=None):
        # if an output function was provided, use it, otherwise use standard one
        if output_fn is None:
            self.output_fn = self.position
            self.output_derivative = self.position_derivative
            self.nderivs = 1 + (state_dim)  # output eval + how many derivatives does that function return
            self.invert_output = self.invert_position
        else:
            self.output_fn = output_fn
            if output_derivative is not None:
                self.output_derivative = output_derivative
        super().__init__(state_dim, 1, f=self.integrator_f, g=self.integrator_g, h=self.output_fn)

    def integrator_f(self, t: float, x: np.ndarray):
        '''
        RHS drift vector field for n-dimensional integrator ODE
        dx[i]/dt = x[i+1]
        dx[n]/dt = u
        '''
        rhs = np.roll(x, -1)
        rhs[-1] = 0.0
        return rhs

    def integrator_g(self, t: float, x: np.ndarray):
        '''
        RHS actuation matrix for n-dimensional integrator ODE
        '''
        return np.eye(self.n, 1, -self.n+1)

    def position(self, t: float, x: np.ndarray, u: np.ndarray):
        '''
        standard output map of just position, i.e., first coordinate
        '''
        return x[0]

    def position_derivative(self, t: float, x: np.ndarray, u: np.ndarray):
        '''
        computes the output and n-1 derivatives at the current state x
        '''
        y_d = np.empty((self.n+1,))
        y_d[:self.n] = x
        y_d[-1] = u
        return y_d

    def invert_position(self, t: float, y_d: np.ndarray, u: np.ndarray):
        '''
        Function that takes the output and its derivatives at time t
        and converts it into the associated state of the n-dimensional integrator
        '''
        return y_d[:self.n]


class LorenzSystem(ControlAffineODE):
    def __init__(self, sigma=10.0, rho=28.0, beta=8.0/3.0, output_fn=None, output_derivative=None):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        if output_fn is None:
            self.output_fn = self.position
            self.output_derivative = self.position_derivative
            self.nderivs = 1 + 3  # output eval + how many derivatives does that function return
            self.invert_output = self.invert_position
            self.TOL = 1e-3
            if output_derivative is not None:
                self.output_derivative = output_derivative
        super().__init__(3, 1, f=self.lorenz_f, h=self.output_fn)

    def lorenz_f(self, t: float, x: np.ndarray, sigma=10.0, rho=28.0, beta=8.0/3.0) -> np.ndarray:
        '''
        RHS for 3-dimensional lorentz attractor
        '''
        rhs = np.array([
            sigma*(x[1]-x[0]),
            x[0]*(rho-x[2])-x[1],
            x[0]*x[1] - beta*x[2]
        ])
        return rhs

    def position(self, t: float, x: np.ndarray, u: np.ndarray):
        return x[0]

    def position_derivative(self, t: float, x: np.ndarray, u: np.ndarray):
        '''
        computes the output and three derivatives of it for the lorenz system
        '''
        y_d = np.empty((4,))
        xdot = self.rhs(t, x, u)
        y_d[0] = x[0]
        y_d[1] = xdot[0]
        y_d[2] = self.sigma*(xdot[1]-xdot[0])
        y_d[3] = (self.sigma*self.rho + self.sigma**2.0)*xdot[0] - (self.sigma + self.sigma**2.0)*xdot[1]
        y_d[3] -= self.sigma*x[0]*xdot[2] - self.sigma*xdot[0]*x[2]
        return y_d

    def invert_position(self, t: float, y_d: np.ndarray, u: np.ndarray):
        '''
        Function that takes the output and its derivatives at time t
        and converts it into the associated state of the lorenz system
        '''
        xhat = np.array([
            y_d[0],
            y_d[0]+y_d[1]/self.sigma,
            -y_d[1]/y_d[0] + self.rho - 1.0 - y_d[1]/(self.sigma*y_d[0]) - y_d[2]/(self.sigma*y_d[0])
        ])
        if np.abs(y_d[0]) < self.TOL:
            xhat[2] = 0.0
        return xhat


class TwoDimExample(ControlAffineODE):
    def __init__(self):
        self.output_fn = self.position
        self.output_derivative = self.position_derivative
        self.nderivs = 1 + 2  # output eval + how many derivatives does that function return
        self.invert_output = self.invert_position
        super().__init__(2, 1, f=self.f, h=self.output_fn)

    def f(self, t: float, x: np.ndarray):
        rhs = np.array([
            -x[0] + x[1]**3.0,
            -x[1]
        ])
        return rhs

    def position(self, t: float, x: np.ndarray, u: np.ndarray):
        return x[0]

    def position_derivative(self, t: float, x: np.ndarray, u: np.ndarray):
        '''
        computes the output and its first 2 derivatives
        '''
        xdot = self.rhs(t, x, u)
        y_d = np.array([
            x[0],
            xdot[0],
            x[0] - 4*x[0]**3.0
        ])
        return y_d

    def invert_position(self, t: float, y: np.ndarray, u: np.ndarray):
        xhat = np.array([
            y[0],
            np.cbrt(y[0]+y[1])
        ])
        return xhat
