from nonlinear_system.ct_system import ContinuousTimeSystem
from nonlinear_system.sample_odes import AckermanModel
from moving_polyfit.moving_ls import MultiDimPolyEstimator
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial as P
import pickle

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42

np.random.seed(0)
verbose = False
##############################################################
#                     TIME  PARAMETERS                       #
##############################################################
N = 51  # number of samples in a window
window_length = 0.5  # number of seconds of trajectory in a single window of data
sampling_dt = window_length/float(N)  # computed sampling timestep

integration_per_sample = 10  # how many integration timesteps should we take between output samples?
integration_dt = sampling_dt/integration_per_sample
num_sampling_steps = 1500  # total number of steps taken in the
num_integration_steps = (num_sampling_steps-1)*integration_per_sample
total_time = num_sampling_steps*sampling_dt

##############################################################
#                    SYSTEM PARAMETERS                       #
##############################################################
noise_mag = 0.075  # 0.025  # magnitude of noise to be applied to outputs
axle_sep = 0.5
wheel_width = 0.6*axle_sep
ODE = AckermanModel(axle_sep, wheel_width)
n = ODE.n  # system state dimension
m = ODE.m  # control input dimension
p = ODE.p  # output dimension
uderivs = ODE.uderivs  # number of control input derivatives we need to provide

print(f'Noise in interval : [{total_time/3.0:.2f}, {total_time*2.0/3.0:.2f}]')


def noise_generator(t: float, mag: float, p: int) -> np.ndarray:
    if total_time/3.0 < t and t < 2*total_time/3:
        return 2*mag*(np.random.rand(p)-0.5)
    else:
        return 0


def wrap_angle(theta: float) -> float:
    return (theta + np.pi) % (2 * np.pi) - np.pi


def control_input(t, y, x=None) -> np.ndarray:
    # if the system has control inputs, we can calculate them here with time-varying output or state feedback
    f = 2.0
    mag = 0.5
    tau = np.pi
    u = np.zeros((2, 3))
    # acceleration and its derivatives
    if t < num_sampling_steps*sampling_dt/3.0:
        u[0, 0] = 2.0
        u[0, 1] = 0.0
        u[0, 2] = 0.0
    elif t > 2*num_sampling_steps*sampling_dt/4.0:
        u[0, 0] = -1.0
    else:
        u[0, 0] = 0.0
        u[0, 1] = 0.0
        u[0, 2] = 0.0

    u[1, 0] = mag*np.cos(f*(t-tau))
    u[1, 1] = -f*mag*np.sin(f*(t-tau))
    u[1, 2] = -(f**2.0)*mag*np.cos(f*(t-tau))
    return u


##############################################################
#                   FITTING PARAMETERS                       #
##############################################################
# currently, we set this to one below the max we can explicitly compute (for bound purposes)
d = ODE.nderivs-1  # degree of estimation polynomial

# this vector will be multiplied with the residuals + noise
l_bound = np.zeros((N, d))

# the theory allows us to pick any subset of (at least) d + 1 points containing the evaluation point.
# we parameterize this by choosing an index jumping size delta
num_t_points = d + 1
delay = N // 2
eval_time = (N-1-delay)*sampling_dt  # (N-1)*sampling_dt
window_times = np.linspace(0., N*sampling_dt, N, endpoint=False)


# TODO: OPTIMIZE DELTA HERE USING MATH
deltas = N//num_t_points
print(f'Optimizing over delta up to: {deltas}')
# this vector will be multiplied with the residuals + noise
l_bound = np.zeros((N, d, deltas))

if num_t_points > N/deltas:
    raise ValueError(f"Delta ({deltas}) invalid for window size ({N}). ({N}/{deltas} = {N/deltas} < {num_t_points})")

verbose_lagrange = False  # to see computation details of lagrange polynomial construction/derivatives

for delta in range(1, deltas+1):
    # for index slicing into the time arrays
    maxstart = N-1-num_t_points*delta
    minstart = 0
    start = np.clip((N-1) - delay - delta*(num_t_points//2), minstart, maxstart)
    l_indices = np.full((num_t_points,), 1)
    for i in range(num_t_points):
        l_indices[i] = start + i*delta
    l_times = window_times[l_indices]  # pull the subset of chosen time indices

    for i in range(num_t_points):
        # build the lagrange polynomial, which is zero at all evaluation samples except one
        evals = np.zeros(num_t_points)
        evals[i] = 1.0  # we are choosing the data points that are closest to our evaluation point
        l_i = P.fit(l_times, evals, d)

        # to checking that you built the right lagrange polynomial, evaluate it at the relevant points
        if verbose_lagrange:
            for j in range(num_t_points):
                print(f't = {l_times[j]:.3f}, l_i(t) = {l_i(l_times[j])}')

        # for every derivative that we estimate, compute this lagrange polynomial's derivative at the estimation time
        for q in range(d):
            l_bound[l_indices[i], q, delta-1] = l_i.deriv(q)(eval_time)  # coefficient for i-th residual in bound
            if verbose_lagrange:
                print(f'|l_{l_indices[i]}^({q})(t)|: {l_bound[l_indices[i], q, delta-1]}')

poly_estimator = MultiDimPolyEstimator(p, d, N, sampling_dt)
global_thetas = False  # if true, computes the coefficients in global time rather than local time frames

sim_sys = ContinuousTimeSystem(ODE, dt=integration_dt, solver='RK45')

# setting up arrays to hold the state, output, and output derivatives at all integration times
x = np.empty((n, num_integration_steps))
y = np.empty((p, num_integration_steps))
y_derivs = np.empty((p, ODE.nderivs, num_integration_steps))

# setting up arrays for errors and bounds in estimation
residual = np.empty((p, N, num_sampling_steps))
cand_bounds = np.zeros((p, d, num_sampling_steps, deltas))
bounds = np.zeros((p, d, num_sampling_steps))
xhat_upper = np.zeros((n, num_sampling_steps))
xhat_lower = np.zeros((n, num_sampling_steps))

# setting up arrays for variables at sampling instants (could be done with index splicing)
y_samples = np.empty((p, num_sampling_steps))
noise_samples = np.empty((p, num_sampling_steps))
y_derivs_samples = np.empty((p, ODE.nderivs, num_sampling_steps))
x_samples = np.empty((n, num_sampling_steps))
u = np.empty((m, uderivs, num_sampling_steps-1))

# estimation and paraemters at sampling times
theta_poly = np.empty((p, d+1, num_sampling_steps))
yhat_poly = np.empty((p, d+1, num_sampling_steps))
xhat_poly = np.empty((n, num_sampling_steps))

# time
integration_time = np.zeros((num_integration_steps,))
sampling_time = np.zeros((num_sampling_steps,))

# initializing the ODE
x0 = 5.0*(np.random.rand(n)-0.5)
x0[0] = -1.0
x0[1] = 0.0
x0[2] = -np.pi/5.0  # np.clip(x0[2], -np.pi, np.pi)
x0[3] = 1.0
x0[4] = 0.0  # np.pi/4.0  # np.clip(x0[4], -np.pi, np.pi)
x[:, 0] = x0
x_samples[:, 0] = x0
sys = ContinuousTimeSystem(ODE, x0=x0, dt=integration_dt, solver='RK45')
y[:, 0] = sys.y
y_samples[:, 0] = sys.y

# SIMULATION LOOP
for t in range(1, num_sampling_steps):
    # select the control input
    u[:, :, t-1] = control_input(sys.t, y[:, t-1], x=x[:, t-1])
    # integrate forward in time
    for i in range(integration_per_sample):
        # compute the right "in between" index
        idx = (t-1)*integration_per_sample + i
        x[:, idx], y[:, idx] = sys.step(u[:, 0, t-1])  # stepping only requires input value
        x[2, idx], x[4, idx] = wrap_angle(x[2, idx]), wrap_angle(x[4, idx])  # manually wrapping angles
        # TODO: fix angle wrapping in CT SYSTEM object
        y_derivs[:, :, idx] = sys.ode.output_derivative(sys.t, sys.x, u[:, :, t-1])  # output derivative ground truth
        integration_time[idx] = sys.t

    # sample the system
    sampling_time[t] = sys.t
    y_samples[:, t], x_samples[:, t] = sys.y, sys.x
    x_samples[2, t], x_samples[4, t] = wrap_angle(x_samples[2, t]), wrap_angle(x_samples[4, t])
    noise_samples[:, t] = noise_generator(sys.t, noise_mag, p)  # generate some noise
    y_samples[:, t] += noise_samples[:, t]  # add it to the signal
    y_derivs_samples[:, :, t] = sys.ode.output_derivative(sys.t, sys.x, u[:, :, t-1])

    if t >= N-1:
        # fit polynomial, save residuals
        theta_poly[:, :, t-delay] = poly_estimator.fit(y_samples[:, t-N+1:t+1])
        residual[:, :, t-delay] = poly_estimator.residuals

        # estimate with polynomial derivatives at endpoint
        for i in range(d+1):
            yhat_poly[:, i, t-delay] = poly_estimator.differentiate(eval_time, i)

        # compute a state estimate from the derivatives
        xhat_poly[:, t-delay] = sys.ode.invert_output(sys.t, yhat_poly[:, :, t-delay], u[:, :, t-1-delay])

        # compute a bound on derivative estimation error from residuals
        for q in range(d):
            for r in range(p):
                noise_vector = np.ones(N,)*noise_mag
                # noise_vector = np.abs(noise_samples[r, t-N+1:t+1])
                for delta in range(deltas):
                    cand_bounds[r, q, t-delay, delta] = np.abs(np.dot(residual[r, :, t-delay], l_bound[:, q, delta]))
                    cand_bounds[r, q, t-delay, delta] += np.dot(noise_vector, np.abs(l_bound[:, q, delta]))

    else:
        theta_poly[:, :, t] = 0.0
        yhat_poly[:, :, t] = 0.0
        residual[:, :, t] = 0.0
        bounds[:, :, t] = 0

    if verbose:
        print(f'Completed timestep {t}, t = {sys.t:.1e}, state = {sys.x}')

M = np.zeros((p,))
for i in range(p):
    M[i] = np.max(np.abs(y_derivs[i, min(ODE.nderivs-1, d), :]))

global_bounds = np.empty((p, d))
for q in range(d):
    for r in range(p):
        global_bounds[r, q] = (M[r]/(np.math.factorial(d+1)))*(np.sqrt(N**2+N))*((N*sampling_dt)**(d+1))
        global_bounds *= np.max(l_bound[:, q])
        global_bounds[r, q] += (M[r]/(np.math.factorial(d-q+1)))*(((q+1)*sampling_dt)**(d-q+1))
        comb = np.math.factorial(d)//(np.math.factorial(d-q+1)*np.math.factorial(max(0, q-1)))
        for delta in range(1, deltas+1):
            cand_bounds[r, q, :, delta-1] += M[r]*comb*((delta*sampling_dt)**(d-q+1))

bounds = np.min(cand_bounds, axis=-1)

for t in range(N-1-delay, num_sampling_steps):
    xhat_upper[0, t] = xhat_poly[0, t] + bounds[0, 0, t]
    xhat_lower[0, t] = xhat_poly[0, t] - bounds[0, 0, t]

    xhat_upper[1, t] = xhat_poly[1, t] + bounds[1, 0, t]
    xhat_lower[1, t] = xhat_poly[1, t] - bounds[1, 0, t]

    xhat_upper[2, t] = np.arctan2(yhat_poly[1, 1, t] + bounds[1, 1, t], yhat_poly[0, 1, t] - bounds[0, 1, t])
    xhat_lower[2, t] = np.arctan2(yhat_poly[1, 1, t] - bounds[1, 1, t], yhat_poly[0, 1, t] + bounds[0, 1, t])

    xhat_upper[3, t] = xhat_poly[3, t] + np.linalg.norm(bounds[:2, 1, t])
    xhat_lower[3, t] = xhat_poly[3, t] - np.linalg.norm(bounds[:2, 1, t])

    # this step does an exhaustive search through upper and lower bound combinations for the last component
    # it's not ideal but requires 16 computations in this case
    yhat_poly_ul = [yhat_poly[:, :4, t] - bounds[:, :, t], yhat_poly[:, :4, t] + bounds[:, :, t]]
    lb = np.inf
    ub = -np.inf
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    test = yhat_poly[:, :, t].copy()
                    test[0, 1] = yhat_poly_ul[i][0, 1].copy()
                    test[1, 1] = yhat_poly_ul[j][1, 1].copy()
                    test[0, 2] = yhat_poly_ul[k][0, 2].copy()
                    test[1, 2] = yhat_poly_ul[l][1, 2].copy()
                    val = sys.ode.invert_output(0.0, test, u[:, :, t-1])[4]
                    lb = min(lb, val)
                    ub = max(ub, val)
    xhat_lower[4, t] = lb
    xhat_upper[4, t] = ub

# SAVING THE DATA HERE
savedata = {'x': x_samples,
            'xhat': xhat_poly,
            'xhat_lower': xhat_lower,
            'xhat_upper': xhat_upper,
            'sim_params': {'dt': sampling_dt,
                           'num_steps': num_sampling_steps,
                           'window_len': N,
                           'axle_sep': axle_sep,
                           'wheel_sep': wheel_width,
                           'delay': delay}}

filename = './tmp/last_run.p'
pickle.dump(savedata, open(filename, 'wb'))

gridrows = 2  # int(np.ceil(d/4))
gridcols = 2  # min(4, d)
size = (5*gridcols, 5)
S = N
E = num_sampling_steps-delay

dest = './tmp/ackerman_'

f4, axs = plt.subplots(nrows=2, ncols=3, figsize=size)
for i in range(n):
    ax = axs.ravel()[i]
    ax.scatter(sampling_time, x_samples[i, :], s=10, marker='x', c='blue', label='samples')
    ax.plot(integration_time, x[i, :], linewidth=2.0, c='blue', label='truth')
    ax.plot(sampling_time[S:E], xhat_poly[i, S:E], linewidth=2.0, c='red',
            linestyle='dashed', label='Polynomial estimate')
    ax.fill_between(sampling_time[S:E], xhat_lower[i, S:E], xhat_upper[i, S:E], color='red', alpha=0.5, zorder=-1)
    ax.set_xlabel('time (s)')
    ax.set_ylabel(f'$x_{i+1}(t)$')
    # ax.legend()
    ax.grid()

f4.axes[-1].set_axis_off()
f4.suptitle('State estimation')
lines_labels = [f4.axes[0].get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
f4.axes[-1].legend(lines, labels, loc="center")
f4.tight_layout()
f4.savefig(dest+'state_estimation.pdf', bbox_inches='tight', pad_inches=0.05)

for q in range(p):
    f5, axs2 = plt.subplots(nrows=gridrows, ncols=gridcols,
                            figsize=size)
    for i, ax in enumerate(axs2.ravel()):
        ax.fill_between(sampling_time[S:E], yhat_poly[q, i, S:E]-bounds[q, i, S:E],
                        yhat_poly[q, i, S:E]+bounds[q, i, S:E],
                        color='red', alpha=0.5, zorder=-1)
        ax.scatter(sampling_time[S:E], yhat_poly[q, i, S:E], color='red', marker='.')
        ax.plot(sampling_time[S:E], yhat_poly[q, i, S:E], linewidth=2.0, c='red',
                linestyle='dashed', label='Polynomial estimate')
        ax.plot(integration_time, y_derivs[q, i, :], linewidth=2.0, c='blue', label='truth')
        ax.set_xlabel('time (s)')
        ax.set_ylabel(f'$y_{q+1}'+"'"*i+'(t)$')
        ax.legend()
        ax.grid()
    f5.suptitle(f'$y_{q+1}$ Derivative estimation')
    f5.tight_layout()
    f5.savefig(dest+f'y_{q+1}_derivative_est.pdf', bbox_inches='tight', pad_inches=0.05)

    f6, axs3 = plt.subplots(nrows=gridrows, ncols=gridcols,
                            figsize=size)
    for i, ax in enumerate(axs3.ravel()):
        ax.plot(sampling_time[S:E], np.abs(yhat_poly[q, i, S:E]-y_derivs_samples[q, i, S:E]), linewidth=2.0,
                c='Blue', label='Polynomial error')
        ax.fill_between(sampling_time[S:E], np.zeros_like(sampling_time[S:E]), bounds[q, i, S:E],
                        alpha=0.5, zorder=-1)
        # ax.semilogy(sampling_time[S:E], np.ones_like(sampling_time[S:E])*global_bounds[q, i], linewidth=2.0,
        #             c='black', label='Offline bound')
        ax.plot(sampling_time[S:E], bounds[q, i, S:E], linewidth=2.0,
                c='red', linestyle='dashed', label='Online bound')
        # ax.plot(integration_time, y_derivs[i, :], linewidth=2.0, c='blue', label='truth')
        ax.set_xlabel('time (s)')
        ax.set_ylabel(f'$y_{q+1}'+"'"*i+'(t) error$')
        ax.legend()
        ax.grid()
    f6.suptitle(f'$y_{q+1}$ Derivative estimation errors')
    f6.tight_layout()
    f6.savefig(dest+f'y_{q+1}_derivative_errors.pdf', bbox_inches='tight', pad_inches=0.05)


plt.show()
