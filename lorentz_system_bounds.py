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
N = 21  # number of samples in a window
window_length = 0.21  # number of seconds of trajectory in a single window of data
sampling_dt = window_length/float(N)  # computed sampling timestep

integration_per_sample = 10  # how many integration timesteps should we take between output samples?
integration_dt = sampling_dt/integration_per_sample
num_sampling_steps = 500  # total number of steps taken in the
num_integration_steps = (num_sampling_steps-1)*integration_per_sample
total_time = num_sampling_steps*sampling_dt

##############################################################
#                    SYSTEM PARAMETERS                       #
##############################################################
noise_mag = 0.5  # magnitude of noise to be applied to outputs

ODE = LorenzSystem()
n = ODE.n
m = ODE.m
p = ODE.p


def noise_generator(t: float, mag: float, p: int) -> np.ndarray:
    if total_time/3.0 < t and t < 2*total_time/3:
        return 2*mag*(np.random.rand(p)-0.5)
    else:
        return 0


def control_input(t, y, x=None):
    # if the system has control inputs, we can calculate them here with time-varying output or state feedback
    return np.array([np.cos(50*t)])  # two_dim_output_deriv(t, x, None)[1]


##############################################################
#                   FITTING PARAMETERS                       #
##############################################################
# currently, we set this to one below the max we can explicitly compute (for bound purposes)
d = ODE.nderivs-1  # degree of estimation polynomial

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
cand_bounds = np.zeros((d, num_sampling_steps, deltas))

# setting up arrays for variables at sampling instants (could be done with index splicing)
y_samples = np.empty((p, num_sampling_steps))
noise_samples = np.empty((p, num_sampling_steps))
y_derivs_samples = np.empty((ODE.nderivs, num_sampling_steps))
x_samples = np.empty((n, num_sampling_steps))
u = np.empty((m, num_sampling_steps-1))

# estimation and paraemters at sampling times
theta_poly = np.empty((d+1, num_sampling_steps))
yhat_poly = np.empty((d+1, num_sampling_steps))
xhat_poly = np.empty((n, num_sampling_steps))
xhat_upper = np.zeros((n, num_sampling_steps))
xhat_lower = np.zeros((n, num_sampling_steps))

# time
integration_time = np.zeros((num_integration_steps,))
sampling_time = np.zeros((num_sampling_steps,))

# initializing the ODE
x0 = 10.0*(np.random.rand(n)-0.5)
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
    noise_samples[:, t] = noise_generator(sampling_time[t], noise_mag, p)  # generate some noise
    y_samples[:, t] += noise_samples[:, t]  # add it to the signal
    y_derivs_samples[:, t] = sys.ode.output_derivative(sys.t, sys.x, u[:, t-1])

    if t >= N-1:
        # fit polynomial, save residuals
        theta_poly[:, t-delay] = poly_estimator.fit(y_samples[0, t-N+1:t+1])
        residual[:, t-delay] = poly_estimator.residuals
        if verbose:
            print(f'T: {sys.t}, Residuals: {np.max(poly_estimator.residuals)}')

        # estimate with polynomial derivatives at endpoint
        for i in range(d+1):
            yhat_poly[i, t-delay] = poly_estimator.differentiate(eval_time, i)

        # compute a state estimate from the derivatives
        xhat_poly[:, t-delay] = sys.ode.invert_output(sys.t, yhat_poly[:, t-delay], u[:, t-1-delay])

        # compute a bound on derivative estimation error from residuals
        noise_vector = np.ones(N,)*noise_mag
        # noise_vector = np.abs(noise_samples[:, t-N+1:t+1])
        for q in range(d):
            for delta in range(deltas):
                cand_bounds[q, t-delay, delta] = np.abs(np.dot(residual[:, t-delay], l_bound[:, q, delta]))
                cand_bounds[q, t-delay, delta] += np.dot(noise_vector, np.abs(l_bound[:, q, delta]))

    else:
        theta_poly[:, t] = 0.0
        yhat_poly[:, t] = 0.0
        residual[:, t] = 0.0
        bounds[:, t] = 0.0

    if verbose:
        print(f'Completed timestep {t}, t = {sys.t:.1e}, state = {sys.x}')

M = np.max(np.abs(y_derivs[min(ODE.nderivs-1, d), :]))
global_bounds = np.empty((d,))
for q in range(d):
    global_bounds[q] = (M/(np.math.factorial(d+1)))*(np.sqrt(N**2+N))*((N*sampling_dt)**(d+1))*np.max(l_bound[:, q, -1])
    global_bounds[q] += np.linalg.norm(noise_vector, 1)*np.max(l_bound[:, q, -1])
    global_bounds[q] += (M/(np.math.factorial(d-q+1)))*(((q+1)*sampling_dt)**(d-q+1))

    # the factorial expression is equivalent to math.comb(d, max(0, q-1))
    comb = np.math.factorial(d)//(np.math.factorial(d-q+1)*np.math.factorial(max(0, q-1)))
    for delta in range(1, deltas+1):
        cand_bounds[q, :, delta-1] += M*comb*((delta*sampling_dt)**(d-q+1))
    #  bounds[q, :] += (M/(np.math.factorial(d-q+1)))*(((q+1)*delta*sampling_dt)**(d-q+1))

bounds = np.min(cand_bounds, axis=-1)  # take best case over all deltas
Ddelta = delay
Nn = d
Tt = sampling_dt
Ssigma = noise_mag
Ee = M

old_bound = 3*Ddelta*(Ddelta+1)*Ee*Tt/(4*(2*Ddelta+1)) + 3*Ssigma/(Tt*(2*Ddelta + 1))

# ONLY FOR LORENTZ SYSTEM
for t in range(N-1-delay, num_sampling_steps):
    bval = np.zeros(d+1)
    bval[0] = bounds[0, t]
    xhat_upper[0, t] = sys.ode.invert_output(sampling_time[t], yhat_poly[:, t]+bval, None)[0]
    xhat_lower[0, t] = sys.ode.invert_output(sampling_time[t], yhat_poly[:, t]-bval, None)[0]

    bval[1] = bounds[1, t]
    xhat_upper[1, t] = sys.ode.invert_output(sampling_time[t], yhat_poly[:, t]+bval, None)[1]
    xhat_lower[1, t] = sys.ode.invert_output(sampling_time[t], yhat_poly[:, t]-bval, None)[1]

    bval[0] = -bounds[0, t]
    bval[1] = bounds[1, t]
    bval[2] = bounds[2, t]
    xhat_upper[2, t] = sys.ode.invert_output(sampling_time[t], yhat_poly[:, t]-bval, None)[2]
    xhat_lower[2, t] = sys.ode.invert_output(sampling_time[t], yhat_poly[:, t]+bval, None)[2]


S = N
E = num_sampling_steps-delay
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
dest = './tmp/lorentz_'

f4, axs = plt.subplots(nrows=n//4+1, ncols=min(4, n), figsize=(5*min(4, n), 5))
for i, ax in enumerate(axs.ravel()):
    ax.scatter(sampling_time, x_samples[i, :], s=20, marker='x', c='blue', label='samples')
    ax.plot(integration_time, x[i, :], linewidth=2.0, c='blue', label='True state')
    ax.fill_between(sampling_time[S:E], xhat_lower[i, S:E], xhat_upper[i, S:E], color='red', alpha=0.5, zorder=-1)
    ax.plot(sampling_time[S:E], xhat_poly[i, S:E], linewidth=2.0, c='red', linestyle='dashed', label='Estimator')
    ax.set_xlabel('time (s)')
    ax.set_ylabel(f'x[{i+1}](t)')
    ax.legend()
    if i == 2:
        ax.set_ylim([-175, 175])
    ax.grid()
f4.suptitle('State estimates')
f4.tight_layout()
f4.savefig(dest+'state_est.pdf', bbox_inches='tight', pad_inches=0.05)

f5, axs2 = plt.subplots(nrows=d//4+1, ncols=min(4, d),
                        figsize=(5*min(4, d), 5))
for i, ax in enumerate(axs2.ravel()):
    ax.fill_between(sampling_time[S:E], yhat_poly[i, S:E]-bounds[i, S:E], yhat_poly[i, S:E]+bounds[i, S:E],
                    color='red', alpha=0.5, zorder=-1)
    ax.scatter(sampling_time[S:E], yhat_poly[i, S:E], color='red', marker='.')
    # ax.errorbar(sampling_time[S:E], yhat_poly[i, S:E], yerr=bounds[i, S:E], color='red')
    ax.plot(integration_time, y_derivs[i, :], linewidth=2.0, c='blue', label='True output')
    ax.plot(sampling_time[S:E], yhat_poly[i, S:E], linewidth=2.0, c='red', linestyle='dashed',
            label='Polynomial estimate')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('y' + "'"*i + '(t)')
    ax.legend()
    ax.grid()
f5.suptitle('Derivative estimates')
f5.tight_layout()
f5.savefig(dest+'derivative_est.pdf', bbox_inches='tight', pad_inches=0.05)

f6, axs3 = plt.subplots(nrows=2, ncols=3,
                        figsize=(5*3, 5), sharex=True)
f6.subplots_adjust(hspace=0.01)

for i in range(3):
    axs3[1, i].plot(sampling_time[S:E], np.abs(yhat_poly[i, S:E]-y_derivs_samples[i, S:E]), linewidth=2.0,
                    c='blue', label='Polynomial error')
    axs3[1, i].fill_between(sampling_time[S:E], np.zeros_like(sampling_time[S:E]), bounds[i, S:E],
                            alpha=0.5, zorder=-1)
    axs3[0, i].plot(sampling_time[S:E], np.ones_like(sampling_time[S:E])*global_bounds[i], linewidth=2.0,
                    c='black', label='Offline bound (Taylor)')
    axs3[1, i].plot(sampling_time[S:E], bounds[i, S:E],
                    linewidth=2.0, c='red', linestyle='dashed', label='Online bound')
    # ax.plot(integration_time, y_derivs[i, :], linewidth=2.0, c='blue', label='truth')
    # if i == 1:
    #     ax.semilogy(sampling_time[S:E], np.ones_like(sampling_time[S:E])*old_bound, linewidth=2.0,
    #                 c='gray', label='Offline bound (previous)')
    axs3[1, i].set_xlabel('time (s)')
    axs3[1, i].set_ylabel('y'+"'"*i + '(t) error')
    axs3[1, i].legend()
    axs3[0, i].legend()
    axs3[1, i].grid()
    axs3[0, i].grid()
    axs3[0, i].spines.bottom.set_visible(False)
    axs3[1, i].spines.top.set_visible(False)
    axs3[0, i].xaxis.tick_top()
    axs3[0, i].tick_params(labeltop=False)
    axs3[1, i].xaxis.tick_bottom()
    axs3[1, i].set_ylim(-0.1*np.max(bounds[i, S:E]), 1.1*np.max(bounds[i, S:E]))
    axs3[0, i].set_ylim(0.9*global_bounds[i], 1.1*global_bounds[i])

    prop = .5  # proportion of vertical to horizontal extent of the slanted line
    slantargs = dict(marker=[(-1, -prop), (1, prop)], markersize=12,
                     linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    axs3[0, i].plot([0, 1], [0, 0], transform=axs3[0, i].transAxes, **slantargs)
    axs3[1, i].plot([0, 1], [1, 1], transform=axs3[1, i].transAxes, **slantargs)
f6.suptitle('Derivative estimation errors')
f6.tight_layout()
f6.savefig(dest+'derivative_error.pdf', bbox_inches='tight', pad_inches=0.05)

f7, axs4 = plt.subplots(nrows=d//4+1, ncols=min(4, d),
                        figsize=(5*min(4, d), 5))
for i, ax in enumerate(axs4.ravel()):
    ax.plot(sampling_time[S:E], np.abs(xhat_poly[i, S:E]-x_samples[i, S:E]), linewidth=2.0,
            c='blue', label='Estimator error')
    ax.fill_between(sampling_time[S:E], np.zeros_like(sampling_time[S:E]),
                    np.maximum(np.abs(xhat_upper[i, S:E]-x_samples[i, S:E]),
                               np.abs(xhat_lower[i, S:E]-x_samples[i, S:E])),
                    alpha=0.5, zorder=-1)
    # ax.plot(sampling_time[S:E], np.ones_like(sampling_time[S:E])*global_bounds[i], linewidth=2.0,
    #             c='black', label='Offline bound (Taylor)')
    ax.plot(sampling_time[S:E], np.maximum(np.abs(xhat_upper[i, S:E]-x_samples[i, S:E]),
                                           np.abs(xhat_lower[i, S:E]-x_samples[i, S:E])),
            linewidth=2.0, c='red', linestyle='dashed', label='Online bound')
    # ax.plot(integration_time, y_derivs[i, :], linewidth=2.0, c='blue', label='truth')
    ax.set_xlabel('time (s)')
    ax.set_ylabel(f'x[{i+1}] error')
    if i == 2:
        ax.set_ylim(bottom=-13, top=300)
    ax.legend()
    ax.grid()
f7.suptitle('State estimation error')
f7.tight_layout()
f7.savefig(dest+'state_error.pdf', bbox_inches='tight', pad_inches=0.05)

plt.show()
