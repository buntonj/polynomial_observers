import numpy as np


def integrator_rhs(t: float, x: np.ndarray, u: float):
    '''
    RHS for n-dimensional integrator ODE
    dx[i]/dt = x[i+1]
    dx[n]/dt = u
    '''
    rhs = np.roll(x, -1)
    rhs[-1] = u
    return rhs


def integrator_output_inv(t: float, y: np.ndarray, u: float, n: int):
    '''
    Function that takes the output and its derivatives at time t
    and converts it into the associated state of the n-dimensional integrator
    '''
    return y[:n]


def lorenz_rhs(t, x, u, sigma=10.0, rho=28.0, beta=8.0/3.0) -> np.ndarray:
    '''
    RHS for 3-dimensional lorentz attractor
    '''
    rhs = np.array([
        sigma*(x[1]-x[0]),
        x[0]*(rho-x[2])-x[1],
        x[0]*x[1] - beta*x[2]
    ])
    return rhs


TOL = 1e-2


def lorenz_output_inv(t: float, y: np.ndarray, u: np.ndarray, sigma=10.0, rho=28.0, beta=8.0/3.0):
    '''
    Function that takes the output and its derivatives at time t
    and converts it into the associated state of the lorenz system
    '''
    xhat = np.array([
        y[0],
        y[0]+y[1]/sigma,
        -y[1]/y[0] + rho - 1.0 - y[1]/(sigma*y[0]) - y[2]/(sigma*y[0])
    ])
    if np.abs(y[0]) < TOL:
        xhat[2] = 0.0
    return xhat


def lorenz_output_deriv(t, x, u):
    '''
    computes the first four derivatives of the output for the lorenz system
    '''
    y = np.empty((4,))
    xdot = lorenz_rhs(t, x, u)
    sigma = 10.0
    rho = 28.0
    beta = 8.0/3.0
    y[0] = x[0]
    y[1] = xdot[0]
    y[2] = sigma*(xdot[1]-xdot[0])
    y[3] = (sigma*rho + sigma**2.0)*xdot[0] - (sigma + sigma**2.0)*xdot[1] - sigma*x[0]*xdot[2] - sigma*xdot[0]*x[2]
    return y


def tora_rhs(t, x, u):
    '''
    RHS for 4-dimensional TORA system
    '''
    k = 1.0  # spring constant
    m1 = 1.0  # mass of carte
    r = 1.0  # length of pendulum rod
    m2 = 1.0  # mass attached to pendulum
    rhs = np.array([
        x[2]/(m1 + m2),
        x[3],
        -k*x[0] + (k*m2*r*np.sin(x[1]))/(m1+m2),
        u
    ])
    return rhs