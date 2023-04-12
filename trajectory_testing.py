from nonlinear_system.ct_system import ContinuousTimeSystem
from nonlinear_system.sample_odes import integrator_rhs, lorenz_rhs, tora_rhs
from nonlinear_system.sample_odes import integrator_output_inv, lorenz_output_inv
from moving_polyfit.moving_ls import PolyEstimator, TrajectoryEstimator
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)
N = 10  # number of samples in a window
window_length = 0.01  # number of seconds of trajectory in a single window of data
sampling_dt = window_length/float(N)  # computed sampling timestep
integration_per_sample = 100  # how many integration timesteps should we take between output samples?
integration_dt = sampling_dt/integration_per_sample
num_sampling_steps = 500  # total number of steps taken in the
num_integration_steps = (num_sampling_steps-1)*integration_per_sample

mid_t = 0.0  # sampling_dt*num_sampling_steps/2.0
f = np.pi/sampling_dt
mag = 1.0
verbose = False


def output_fn(t, x, u):
    return x[0]


def control_input(t, y):
    return -y[0]**3.0


def generate_simulated_trajectories(sys, num, N, dt):
    trajectories = []
    for i in range(num):
        x = np.empty((sys.n, N))
        y = np.empty((sys.p, N))
        x0 = 50.0*(np.random.rand(sys.n) - 0.5)
        x[:, 0], y[:, 0] = sys.reset(x0, u0=None, t=0.0)
        for t in range(N):
            u = control_input(t*dt, y[:, 0])
            x[:, t], y[:, t] = sys.step(u, dt=dt)
        trajectories.append((x, y))
    return trajectories


sigma = 0.1
ODE_RHS = lambda t, x, u: lorenz_rhs(t, x, u, sigma=0.1)
OUTPUT_INV = lambda t, y, u: lorenz_output_inv(t, y, u, sigma=0.1)
n = 3  # system state dimension
m = 1  # control input dimension
p = 1  # output dimension

d = 5  # degree of estimation polynomial
poly_estimator = PolyEstimator(d, N, sampling_dt)
# residual_poly = PolyEstimator(d, d, sampling_dt)
global_thetas = False

num_trajectories = 6
sim_sys = ContinuousTimeSystem(n, ODE_RHS, h=output_fn, dt=integration_dt, solver='RK45')
trajectories = generate_simulated_trajectories(sim_sys, num_trajectories, N, sampling_dt)
print('GENERATED TRAJECTORIES.')
traj_estimator = TrajectoryEstimator(trajectories)
state_fit = TrajectoryEstimator(trajectories)
# print('REGRESSION MATRICES:')
# print(f'Output: {traj_estimator.regression_matrix}')
# print(f'State: {traj_estimator.state_regression_matrix}')
C_matrix = np.zeros((N, n*N))
for i in range(N):
    C_matrix[i, i*n] = 1.0
# print(f'Cphi: {np.linalg.pinv(C_matrix @ traj_estimator.state_data_matrix)}')
# print(f'C+C: {np.linalg.pinv(C_matrix) @ C_matrix}')

x = np.empty((n, num_integration_steps))
y = np.empty((p, num_integration_steps))
residual = np.empty((d, num_integration_steps))

y_samples = np.empty((p, num_sampling_steps))
x_samples = np.empty((n, num_sampling_steps))
u = np.empty((m, num_sampling_steps-1))

theta_poly = np.empty((d+1, num_sampling_steps))
theta_traj = np.empty((num_trajectories, num_sampling_steps))
theta_state_traj = np.empty((num_trajectories, num_sampling_steps))
yhat_poly = np.empty((d+1, num_sampling_steps))
xhat_poly = np.empty((n, num_sampling_steps))
xhat_traj = np.empty((n, num_sampling_steps))

integration_time = np.zeros((num_integration_steps,))
sampling_time = np.zeros((num_sampling_steps,))
x0 = 50.0*(np.random.rand(n)-0.5)
x[:, 0] = x0
x_samples[:, 0] = x0
sys = ContinuousTimeSystem(n, ODE_RHS, h=output_fn, x0=x0, dt=integration_dt, solver='RK45')
y[:, 0] = sys.y
y_samples[0, 0] = sys.y

print('ESTIMATOR PROPERTIES:')
print(f'Traject condition number: {np.linalg.cond(traj_estimator.data_matrix)}')
_, s, _ = np.linalg.svd(traj_estimator.data_matrix)
print('Singular values', s)


for t in range(1, num_sampling_steps):
    # select the control input
    u[:, t-1] = control_input(sys.t, y[:, t-1])
    # integrate forward in time
    for i in range(integration_per_sample):
        # compute the right "in between" index
        idx = (t-1)*integration_per_sample + i
        x[:, idx], y[:, idx] = sys.step(u[:, t-1])
        integration_time[idx] = sys.t
        # if t > N-1:
        #    residual[:, idx] = y[:, idx] - residual_poly.estimate(sys.t)

    # sample the system
    sampling_time[t] = sys.t
    y_samples[0, t], x_samples[:, t] = sys.y, sys.x

    if t >= N-1:
        # POLYNOMIAL FITTING
        if global_thetas:
            # fit polynomial
            theta_poly[:, t] = poly_estimator.fit(y_samples[0, t-N+1:t+1], sampling_time[t-N+1])

            # estimate with polynomial derivatives at endpoint
            for i in range(d+1):
                yhat_poly[i, t] = poly_estimator.differentiate(sampling_time[t], i)
        else:
            # fit polynomial
            theta_poly[:, t] = poly_estimator.fit(y_samples[0, t-N+1:t+1])

            # estimate with polynomial derivatives at endpoint
            for i in range(d+1):
                yhat_poly[i, t] = poly_estimator.differentiate((N-1)*sampling_dt, i)
        xhat_poly[:, t] = OUTPUT_INV(sys.t, yhat_poly[:, t], u[:, t-1])
        #residual_poly.fit(y_samples[0, t-N+1:t-N+d+1]-yhat_poly[0, t-N+1:t-N+d+1])

        # TRAJECTORY FITTING
        theta_traj[:, t] = traj_estimator.fit(y_samples[0, t-N+1:t+1])
        theta_state_traj[:, t] = state_fit.fit_state(x_samples[:, t-N+1:t+1])
        xhat_traj[:, t] = traj_estimator.estimate()

    else:
        theta_state_traj[:, t] = 0.0
        theta_poly[:, t] = 0.0
        theta_traj[:, t] = 0.0
        yhat_poly[:, t] = 0.0
        xhat_traj[:, t] = 0.0
        residual[:, t] = 0.0

    if verbose:
        print(f'Completed timestep {t}, t = {sys.t:.1e}, state = {sys.x}')
        print(f'Difference: {theta_state_traj[:,t]-theta_traj[:,t]}')


f = plt.figure(figsize=(12, 8))
x0_plot = f.add_subplot((221))
x1_plot = f.add_subplot((222))
traj_plot = f.add_subplot((223))
u_plot = f.add_subplot((224))

x0_plot.plot(integration_time, x[0, :], linewidth=2.0, c='blue', label='truth')
x0_plot.plot(sampling_time[N:], xhat_poly[0, N:], linewidth=2.0, c='red', linestyle='dashed', label='poly estimate')
x0_plot.plot(sampling_time[N:], xhat_traj[0, N:], linewidth=2.0, c='green', linestyle='dashed', label='traj estimate')
x0_plot.scatter(sampling_time, x_samples[0, :], s=20, marker='x', c='blue', label='samples')
x0_plot.set_xlabel('time (s)')
x0_plot.set_ylabel('x1(t)')
x0_plot.legend()
x0_plot.grid()

x1_plot.plot(integration_time, x[1, :], linewidth=2.0, c='blue', label='truth')
x1_plot.plot(sampling_time[N:], xhat_poly[1, N:], linewidth=2.0, c='red', linestyle='dashed', label='poly estimate')
x1_plot.plot(sampling_time[N:], xhat_traj[1, N:], linewidth=2.0, c='green', linestyle='dashed', label='traj estimate')
x1_plot.scatter(sampling_time, x_samples[1, :], s=50, marker='x', c='blue', label='samples')
x1_plot.set_xlabel('time (s)')
x1_plot.set_ylabel('x2(t)')
x1_plot.legend()
x1_plot.grid()

traj_plot.plot(x[0, :], x[1, :], linewidth=2.0, c='blue', label='truth')
traj_plot.plot(xhat_poly[0, N:], xhat_poly[1, N:], linewidth=2.0, c='red', linestyle='dashed', label='poly estimate')
traj_plot.plot(xhat_traj[0, N:], xhat_traj[1, N:], linewidth=2.0, c='green', linestyle='dashed', label='traj estimate')
traj_plot.scatter(x[0, 0], x[1, 0], s=75, marker='*', c='blue')
traj_plot.scatter(xhat_poly[0, N], xhat_poly[1, N], s=20, marker='*', c='red')
traj_plot.scatter(x_samples[0, :], x_samples[1, :], s=20, marker='x', c='blue')
marg = 0.1
traj_plot.set_xlim(x[0, :].min()-marg, x[0, :].max()+marg)
traj_plot.set_ylim(x[1, :].min()-marg, x[1, :].max()+marg)
traj_plot.set_xlabel('x1')
traj_plot.set_ylabel('x2')
traj_plot.legend()
traj_plot.grid()

u_plot.plot(sampling_time[1:], u[0, :], linewidth=2.0, c='red')
u_plot.set_xlabel('time (s)')
u_plot.set_ylabel('u')
u_plot.grid()
u_plot.set_title('Input over time')
f.tight_layout()

f2 = plt.figure(figsize=(15, 6))
thetas_plot = f2.add_subplot(121)

for i in range(d+1):
    thetas_plot.plot(sampling_time[N:], theta_poly[i, N:], linewidth=2.0, label=f'Theta[{i}], poly')
for i in range(n):
    thetas_plot.plot(sampling_time[N:], theta_traj[i, N:], linewidth=2.0, linestyle='dashed', label=f'Theta[{i}], traj')
thetas_plot.set_ylim(min(np.amin(theta_poly[:, N:]), np.amin(theta_traj[:, N:])),
                     max(np.amax(theta_traj[:, N:]), np.amax(theta_poly[:, N:])))
thetas_plot.legend()
thetas_plot.set_title('Parameters over time')
thetas_plot.grid()

yhat_plot = f2.add_subplot(122)
colors = ['red', 'blue', 'green', 'orange', 'pink', 'deeppink', 'violet', 'lime', 'gray', 'black']
for i in range(n):
    kwargs = {
        'label': f'x[{i}] error, poly',
        'linewidth': 2.0,
        'color': colors[i]
    }
    yhat_plot.plot(sampling_time[N:], np.abs(xhat_poly[i, N:]-x_samples[i, N:]), **kwargs)
    kwargs = {
        'label': f'x[{i}] error, traj',
        'linewidth': 2.0,
        'color': colors[i],
        'linestyle': 'dashed'
    }
    yhat_plot.plot(sampling_time[N:], np.abs(xhat_traj[i, N:]-x_samples[i, N:]), **kwargs)
yhat_plot.legend()
yhat_plot.grid()
yhat_plot.set_xlabel('time')
yhat_plot.set_ylabel('error')
yhat_plot.set_title('State estimation errors')
f2.tight_layout()

f3 = plt.figure(figsize=(12, 6))
sample_x_plot = f3.add_subplot(121)
sample_y_plot = f3.add_subplot(122)
for i, trajectory in enumerate(trajectories):
    c = colors[i % len(colors)]
    sample_x_plot.scatter(trajectory[0][0, 0], trajectory[0][1, 0], marker='*', s=50, color=c)
    sample_x_plot.plot(trajectory[0][0, :], trajectory[0][1, :], linewidth=2.0, linestyle='dashed',
                       label=f'traj sample {i}', color=c)
    sample_y_plot.plot(sampling_time[:N], trajectory[1][0, :], linewidth=2.0, linestyle='dashed',
                       label=f'traj sample {i}', color=c)
for i in range(d+1):
    c = colors[i % len(colors)]
    sample_y_plot.plot(sampling_time[:N], sampling_time[:N]**float(i), linewidth=2.0, label=f'poly sample {i}', color=c)
sample_x_plot.grid()
sample_y_plot.grid()
sample_x_plot.set_title('Trajectory Samples')
sample_x_plot.set_xlabel('x[0]')
sample_x_plot.set_ylabel('x[1]')
sample_y_plot.set_xlabel('t')
sample_y_plot.set_ylabel('y(t)')
sample_y_plot.set_title('Output samples')
f3.tight_layout()

f4, axs = plt.subplots(nrows=n//4+1, ncols=min(4, n), figsize=(5*min(4, n), 5))
for i, ax in enumerate(axs.ravel()):
    ax.plot(integration_time, x[i, :], linewidth=2.0, c='blue', label='truth')
    ax.plot(sampling_time[N:], xhat_poly[i, N:], linewidth=2.0, c='red', linestyle='dashed', label='poly estimate')
    ax.plot(sampling_time[N:], xhat_traj[i, N:], linewidth=2.0, c='green', linestyle='dashed', label='traj estimate')
    ax.scatter(sampling_time, x_samples[i, :], s=20, marker='x', c='blue', label='samples')
    ax.set_xlabel('time (s)')
    ax.set_ylabel(f'x[{i}](t)')
    ax.legend()
    ax.grid()
f4.tight_layout()

plt.show()
