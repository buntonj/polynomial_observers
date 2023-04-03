import numpy as np
from scipy.integrate import solve_ivp
# TODO: LTV and LTI system child classes for easier interfaces


class ContinuousTimeSystem:
    '''
    ContinuousTimeSystem object.
    Holds an ODE evolving according to the equations:

    dx/dt = f(t,x,u)
    y(t) = h(t,x,u)

    INPUTS:
    n - system dimension
    rhs - ode right-hand side.
    rhs function handle is rhs(t,x,u) -> vector field as (n,) array
    h - output function function handle, form h(t,x,u) -> y as (p,) array
        if u is None, should return value of output for some generic u (e.g., u = 0)
    t - current time for the system. Default value is 0.0
    dt - timestep for sample-and-hold model, default 1-e6 or 1us
    x0 - initial state of system

    '''
    def __init__(self, n: int, rhs, h=None, x0=None, u0=None, t=0.0, dt=1e-6, solver='RK45'):

        self.n = n
        self.t = t
        self.rhs = rhs
        if h is None:
            self.output = identity_output
        else:
            self.output = h
        self.dt = dt

        if x0 is None:
            self.x = np.zeros((self.n,))
        else:
            self.x = x0

        self.u = u0
        self.y = self.output(self.t, self.x, self.u)
        if self.y.shape == tuple():
            self.p = 1
        else:
            self.p = self.y.shape[0]

        self.solver = solver

    def step(self, u, dt=None):
        # parameterize the ODE right hand side (using a def for cleanliness)
        def f(t, x):
            return self.rhs(t, x, u)
        self.u = u
        if dt is None:
            dt = self.dt
        sol = solve_ivp(f, (self.t, self.t+dt), self.x, method=self.solver)  # numerically integrate for self.dt
        self.t += dt  # step forward the time variable
        self.x = sol.y[:, -1]  # save the new system state
        self.y = self.output(self.t, self.x, u)
        return self.x, self.y

    def reset(self, x0: np.ndarray, u0=None, t=0.0):
        '''
        Resets the system object's state variable and time variable.
        '''
        self.x = x0
        self.t = t
        self.y = self.output(self.t, self.x, self.u)
        return self.x, self.y


def identity_output(t, x, u):
    return x
