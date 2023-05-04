from nonlinear_system.ct_system import ContinuousTimeSystem
from nonlinear_system.sample_odes import LorenzSystem
from moving_polyfit.moving_ls import PolyEstimator
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial as P

np.random.seed(0)
verbose = False
##############################################################
#                     TIME  PARAMETERS                       #
##############################################################
N = 10  # number of samples in a window
window_length = 0.01  # number of seconds of trajectory in a single window of data
sampling_dt = window_length/float(N)  # computed sampling timestep

integration_per_sample = 10  # how many integration timesteps should we take between output samples?
integration_dt = sampling_dt/integration_per_sample
num_sampling_steps = 500  # total number of steps taken in the
num_integration_steps = (num_sampling_steps-1)*integration_per_sample

##############################################################
#                    SYSTEM PARAMETERS                       #
##############################################################
n = 2  # system state dimension
m = 1  # control input dimension
p = 1  # output dimension

ODE = LorenzSystem()
n = ODE.n
m = ODE.m
p = ODE.p


def control_input(t, y, x=None):
    # if the system has control inputs, we can calculate them here with time-varying output or state feedback
    return np.array([np.cos(50*t)])  # two_dim_output_deriv(t, x, None)[1]


##############################################################
#                   FITTING PARAMETERS                       #
##############################################################
# currently, we set this to one below the max we can explicitly compute (for bound purposes)
d = ODE.nderivs-1  # degree of estimation polynomial

# this vector will be multiplied with the residuals
l_bound = np.zeros((N, d))

# the theory allows us to pick any subset of (at least) d + 1 points containing the evaluation point.
# for simplicity, here we default to the last data points.
num_t_points = d + 1
eval_time = (N-1)*sampling_dt
l_times = np.linspace((N-num_t_points)*sampling_dt, N*sampling_dt, num_t_points, endpoint=False)
verbose_lagrange = False  # to see computation details of lagrange polynomial construction/derivatives

for i in range(num_t_points):
    # build the lagrange polynomial, which is zero at all evaluation samples except one
    evals = np.zeros(num_t_points)
    evals[-(1+i)] = 1.0  # we are choosing the data points that are closest to our evaluation point
    l_i = P.fit(l_times, evals, d)

    # to checking that you built the right lagrange polynomial, evaluate it at the relevant points
    if verbose_lagrange:
        for j in range(num_t_points):
            print(f't = {l_times[j]:.3f}, l_i(t) = {l_i(l_times[j])}')

    # for every derivative that we estimate, compute this lagrange polynomial's derivative at the estimation time
    for q in range(d):
        l_bound[-(1+i), q] = np.abs(l_i.deriv(q)(eval_time))  # coefficient for i-th residual in bound
        if verbose_lagrange:
            print(f'|l_{num_t_points-i}^({q})(t)|: {l_bound[-(1+i), q]}')  # for an idea of the scale of each term


poly_estimator = PolyEstimator(d, N, sampling_dt)
global_thetas = False  # if true, computes the coefficients in global time rather than local time frames

sim_sys = ContinuousTimeSystem(ODE, dt=integration_dt, solver='RK45')

# setting up arrays to hold the state, output, and output derivatives at all integration times
x = np.empty((n, num_integration_steps))
y = np.empty((p, num_integration_steps))
y_derivs = np.empty((ODE.nderivs, num_integration_steps))

# setting up arrays for errors and bounds in estimation
residual = np.empty((N, num_sampling_steps))
bounds = np.zeros((d, num_sampling_steps))

# setting up arrays for variables at sampling instants (could be done with index splicing)
y_samples = np.empty((p, num_sampling_steps))
y_derivs_samples = np.empty((ODE.nderivs, num_sampling_steps))
x_samples = np.empty((n, num_sampling_steps))
u = np.empty((m, num_sampling_steps-1))

# estimation and paraemters at sampling times
theta_poly = np.empty((d+1, num_sampling_steps))
yhat_poly = np.empty((d+1, num_sampling_steps))
xhat_poly = np.empty((n, num_sampling_steps))

# time
integration_time = np.zeros((num_integration_steps,))
sampling_time = np.zeros((num_sampling_steps,))

# initializing the ODE
x0 = 5.0*(np.random.rand(n)-0.5)
x[:, 0] = x0
x_samples[:, 0] = x0
sys = ContinuousTimeSystem(ODE, x0=x0, dt=integration_dt, solver='RK45')
y[:, 0] = sys.y
y_samples[0, 0] = sys.y

# SIMULATION LOOP
for t in range(1, num_sampling_steps):
    # select the control input
    u[:, t-1] = control_input(sys.t, y[:, t-1], x=x[:, t-1])
    # integrate forward in time
    for i in range(integration_per_sample):
        # compute the right "in between" index
        idx = (t-1)*integration_per_sample + i
        x[:, idx], y[:, idx] = sys.step(u[:, t-1])
        y_derivs[:, idx] = sys.ode.output_derivative(sys.t, sys.x, u[:, t-1])
        integration_time[idx] = sys.t

    # sample the system
    sampling_time[t] = sys.t
    y_samples[0, t], x_samples[:, t] = sys.y, sys.x
    y_derivs_samples[:, t] = sys.ode.output_derivative(sys.t, sys.x, u[:, t-1])

    if t >= N-1:
        # fit polynomial, save residuals
        theta_poly[:, t] = poly_estimator.fit(y_samples[0, t-N+1:t+1])
        residual[:, t] = poly_estimator.residuals

        # estimate with polynomial derivatives at endpoint
        for i in range(d+1):
            yhat_poly[i, t] = poly_estimator.differentiate((N-1)*sampling_dt, i)

        # compute a state estimate from the derivatives
        xhat_poly[:, t] = sys.ode.invert_output(sys.t, yhat_poly[:, t], u[:, t-1])

        # compute a bound on derivative estimation error from residuals
        for q in range(d):
            bounds[q, t] = np.dot(residual[:, t], l_bound[:, q])

    else:
        theta_poly[:, t] = 0.0
        yhat_poly[:, t] = 0.0
        residual[:, t] = 0.0
        bounds[:, t] = 0

    if verbose:
        print(f'Completed timestep {t}, t = {sys.t:.1e}, state = {sys.x}')

M = np.max(np.abs(y_derivs[min(ODE.nderivs-1, d), :]))
global_bounds = np.empty((d,))
for q in range(d):
    # global_bounds[q] = (M/np.math.factorial(d+1))*np.dot(l_bound[:, q],
    #                                                     np.linspace(0.0, (N-1)*sampling_dt, N, endpoint=True)**(d+1))
    global_bounds[q] = (M/(np.math.factorial(d+1)))*(np.sqrt(N**2+N))*((N*sampling_dt)**(d+1))*np.max(l_bound[:, q])
    global_bounds[q] += (M/(np.math.factorial(d-q+1)))*(((q+1)*sampling_dt)**(d-q+1))
    bounds[q, :] += (M/(np.math.factorial(d-q+1)))*(((q+1)*sampling_dt)**(d-q+1))

f4, axs = plt.subplots(nrows=d//4+1, ncols=min(4, n), figsize=(5*min(4, n), 5))
for i, ax in enumerate(axs.ravel()):
    ax.scatter(sampling_time, x_samples[i, :], s=20, marker='x', c='blue', label='samples')
    ax.plot(sampling_time[N:], xhat_poly[i, N:], linewidth=2.0, c='red', linestyle='dashed', label='poly estimate')
    ax.plot(integration_time, x[i, :], linewidth=2.0, c='blue', label='truth')
    ax.set_xlabel('time (s)')
    ax.set_ylabel(f'x[{i}](t)')
    ax.legend()
    ax.grid()
f4.tight_layout()

f5, axs2 = plt.subplots(nrows=d//4+1, ncols=min(4, n),
                        figsize=(5*min(4, d), 5))
for i, ax in enumerate(axs2.ravel()):
    ax.fill_between(sampling_time[N:], yhat_poly[i, N:]-bounds[i, N:], yhat_poly[i, N:]+bounds[i, N:],
                    color='red', alpha=0.5, zorder=-1)
    ax.scatter(sampling_time[N:], yhat_poly[i, N:], color='red', marker='1')
    # ax.errorbar(sampling_time[N:], yhat_poly[i, N:], yerr=bounds[i, N:], color='red')
    ax.plot(sampling_time[N:], yhat_poly[i, N:], linewidth=2.0, c='red', linestyle='dashed', label='poly estimate')
    ax.plot(integration_time, y_derivs[i, :], linewidth=2.0, c='blue', label='truth')
    ax.set_xlabel('time (s)')
    ax.set_ylabel(f'y^({i})(t)')
    ax.legend()
    ax.grid()
f5.tight_layout()

f6, axs3 = plt.subplots(nrows=d//4+1, ncols=min(4, n),
                        figsize=(5*min(4, d), 5))
for i, ax in enumerate(axs3.ravel()):
    ax.plot(sampling_time[N:], np.abs(yhat_poly[i, N:]-y_derivs_samples[i, N:]), linewidth=2.0,
            c='red', label='poly error')
    ax.fill_between(sampling_time[N:], np.zeros_like(sampling_time[N:]), bounds[i, N:],
                    alpha=0.5, zorder=-1)
    ax.plot(sampling_time[N:], np.ones_like(sampling_time[N:])*global_bounds[i], linewidth=2.0,
            c='black', label='global bound')
    ax.plot(sampling_time[N:], bounds[i, N:], linewidth=2.0, c='red', linestyle='dashed', label='poly bound')
    # ax.plot(integration_time, y_derivs[i, :], linewidth=2.0, c='blue', label='truth')
    ax.set_xlabel('time (s)')
    ax.set_ylabel(f'y^({i})(t)')
    ax.legend()
    ax.grid()
f6.tight_layout()

plt.show()
