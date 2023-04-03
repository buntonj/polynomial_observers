from nonlinear_system.ct_system import ContinuousTimeSystem
from moving_polyfit.moving_ls import PolyEstimator, TrajectoryEstimator
import numpy as np
import matplotlib.pyplot as plt


sampling_dt = 1.0  # sampling timestep
integration_per_sample = 500  # how many integration timesteps should we take between output samples?
integration_dt = sampling_dt/integration_per_sample
num_sampling_steps = 20
num_integration_steps = num_sampling_steps*integration_per_sample

mid_t = 0.0  # sampling_dt*num_sampling_steps/2.0
f = np.pi/sampling_dt
mag = 1.0
verbose = False


def phi(t, x):
    return -1.0  # mag*np.sin(f*t)


def dphidt(t, x):
    return 0.0  # f*mag*np.cos(f*t)


def rhs(t, x, u):
    '''
    RHS for n-dimensional integrator ODE
    dx[i]/dt = x[i+1]
    dx[n]/dt = phi(x)
    '''
    rhs = np.roll(x, -1)
    rhs[-1] = phi(t, x)
    return rhs


def output_fn(t, x, u):
    return x[0]


def control_input(t, y):
    return 0.0


def generate_simulated_trajectories(sys, num, N, dt):
    trajectories = []
    for i in range(num):
        x = np.empty((sys.n, N))
        y = np.empty((sys.p, N))
        x0 = 2.0*(np.random.rand(sys.n) - 0.5)
        x[:, 0], y[:, 0] = sys.reset(x0, u0=None, t=0.0)
        for t in range(N):
            u = control_input(t*dt, y[:, 0])
            x[:, t], y[:, t] = sys.step(u, dt=dt)
        trajectories.append((x, y))
    return trajectories


n = 2  # system state dimension
m = 1  # control input dimension
p = 1  # output dimension

d = 4  # degree of estimation polynomial
N = 10  # number of samples
poly_estimator = PolyEstimator(d, N, sampling_dt)
global_thetas = False

num_trajectories = d
sim_sys = ContinuousTimeSystem(2, rhs, h=output_fn, dt=integration_dt, solver='RK45')
trajectories = generate_simulated_trajectories(sim_sys, num_trajectories, N, sampling_dt)
traj_estimator = TrajectoryEstimator(trajectories)

x = np.empty((n, num_integration_steps))
y = np.empty((p, num_integration_steps))

y_samples = np.empty((d, num_sampling_steps))
x_samples = np.empty((n, num_sampling_steps))
u = np.empty((m, num_sampling_steps-1))

theta_poly = np.empty((d, num_sampling_steps))
theta_traj = np.empty((num_trajectories, num_sampling_steps))
dphi_hat = np.empty((num_sampling_steps,))
yhat_poly = np.empty((d, num_sampling_steps))
yhat_traj = np.empty((n, num_sampling_steps))

y_true = np.zeros((d, num_integration_steps))

integration_time = np.zeros((num_integration_steps,))
sampling_time = np.zeros((num_sampling_steps,))
np.random.seed(2)
x0 = np.array([0.0, -mag/f])
x[:, 0] = x0
x_samples[:, 0] = x0
sys = ContinuousTimeSystem(2, rhs, h=output_fn, x0=x0, dt=integration_dt, solver='RK45')
y[:, 0] = sys.y
y_samples[0, 0] = sys.y
y_true[0, 0] = sys.y

print("Initialized CT system object.")

print('ESTIMATION PROPERTIES:')
print(f'Polyfit condition number: {np.linalg.cond(poly_estimator.F)}')
print(f'Traject condition number: {np.linalg.cond(traj_estimator.data_matrix)}')


for t in range(1, num_sampling_steps):
    # select the control input
    u[:, t-1] = control_input(sys.t, y[:, t-1])

    # integrate forward in time
    for i in range(integration_per_sample):
        # compute the right "in between" index
        idx = t*integration_per_sample + i
        x[:, idx], y[:, idx] = sys.step(u[:, t-1])
        integration_time[idx] = sys.t

    # extract real values of output and its derivatives
        for j in range(sys.n):
            y_true[j, idx] = x[j, idx]
        y_true[sys.n, idx] = phi(sys.t, x[:, idx])
        y_true[sys.n+1, idx] = dphidt(sys.t, x[:, idx])

    # sample the system
    sampling_time[t] = sys.t
    y_samples[0, t], x_samples[:, t] = sys.y, sys.x

    for i in range(sys.n):
        y_samples[i, t] = x_samples[i, t]
    y_samples[sys.n, t] = phi(sys.t, x_samples[:, t])
    y_samples[sys.n+1, t] = dphidt(sys.t, x_samples[:, t])

    if t >= N-1:
        # POLYNOMIAL FITTING
        if global_thetas:
            # fit polynomial
            theta_poly[:, t] = poly_estimator.fit_global(y_samples[0, t-N+1:t+1], sampling_time[t-N+1])

            # estimate with polynomial derivatives at endpoint
            for i in range(d):
                yhat_poly[i, t] = poly_estimator.differentiate(sampling_time[t], i)
        else:
            # fit polynomial
            theta_poly[:, t] = poly_estimator.fit(y_samples[0, t-N+1:t+1])

            # estimate with polynomial derivatives at endpoint
            for i in range(d):
                yhat_poly[i, t] = poly_estimator.differentiate((N-1)*sampling_dt, i)

        # TRAJECTORY FITTING
        theta_traj[:, t] = traj_estimator.fit(y_samples[0, t-N+1:t+1])
        yhat_traj[:, t] = traj_estimator.estimate()

    else:
        theta_poly[:, t] = 0.0
        theta_traj[:, t] = 0.0
        yhat_poly[:, t] = 0.0
        yhat_traj[:, t] = 0.0

    if verbose:
        print(f'Completed timestep {t}, t = {sys.t:.1e}, state = {sys.x}')


f = plt.figure(figsize=(12, 8))
x0_plot = f.add_subplot((221))
x1_plot = f.add_subplot((222))
traj_plot = f.add_subplot((223))
u_plot = f.add_subplot((224))

x0_plot.plot(integration_time, x[0, :], linewidth=2.0, c='blue', label='truth')
x0_plot.plot(sampling_time[N:], yhat_poly[0, N:], linewidth=2.0, c='red', linestyle='dashed', label='poly estimate')
x0_plot.plot(sampling_time[N:], yhat_traj[0, N:], linewidth=2.0, c='green', linestyle='dashed', label='traj estimate')
x0_plot.scatter(sampling_time, y_samples[0, :], s=50, marker='x', c='blue', label='samples')
x0_plot.set_xlabel('time (s)')
x0_plot.set_ylabel('x1(t)')
x0_plot.legend(loc='upper right')
x0_plot.grid()

x1_plot.plot(integration_time, x[1, :], linewidth=2.0, c='blue', label='truth')
x1_plot.plot(sampling_time[N:], yhat_poly[1, N:], linewidth=2.0, c='red', linestyle='dashed', label='poly estimate')
x1_plot.plot(sampling_time[N:], yhat_traj[1, N:], linewidth=2.0, c='green', linestyle='dashed', label='traj estimate')
x1_plot.scatter(sampling_time, y_samples[1, :], s=50, marker='x', c='blue', label='samples')
x1_plot.set_xlabel('time (s)')
x1_plot.set_ylabel('x2(t)')
x1_plot.legend(loc='upper right')
x1_plot.grid()

traj_plot.plot(x[0, :], x[1, :], linewidth=2.0, c='blue', label='truth')
traj_plot.plot(yhat_poly[0, N:], yhat_poly[1, N:], linewidth=2.0, c='red', linestyle='dashed', label='poly estimate')
traj_plot.plot(yhat_traj[0, N:], yhat_traj[1, N:], linewidth=2.0, c='green', linestyle='dashed', label='traj estimate')
traj_plot.scatter(x[0, 0], x[1, 0], s=50, marker='*', c='blue')
traj_plot.scatter(yhat_poly[0, N], yhat_poly[1, N], s=50, marker='*', c='red')
traj_plot.scatter(x_samples[0, :], x_samples[1, :], s=50, marker='x', c='blue')
marg = 0.1
traj_plot.set_xlim(x[0, :].min()-marg, x[0, :].max()+marg)
traj_plot.set_ylim(x[1, :].min()-marg, x[1, :].max()+marg)
traj_plot.set_xlabel('x1')
traj_plot.set_ylabel('x2')
traj_plot.legend(loc='upper right')
traj_plot.grid()

u_plot.plot(sampling_time[1:], u[0, :], linewidth=2.0, c='red')
u_plot.set_xlabel('time (s)')
u_plot.set_ylabel('u')
u_plot.grid()

f.tight_layout()

f2 = plt.figure(figsize=(15, 6))
thetas_plot = f2.add_subplot(131)

for i in range(d):
    thetas_plot.plot(sampling_time[N:], theta_poly[i, N:], linewidth=2.0, label=f'Theta[{i}]')
thetas_plot.set_ylim(-1, 1)
thetas_plot.legend(loc='upper right')
thetas_plot.grid()

est_plot = f2.add_subplot(132)
est_plot.plot(integration_time[N:], y_true[3, N:], linewidth=2.0, c='blue', label='truth')
est_plot.plot(sampling_time[N:], yhat_poly[3, N:], linewidth=2.0, c='red', label='estimate')
est_plot.set_ylabel('d/dt phi(t)')
est_plot.legend(loc='upper right')
est_plot.grid()

yhat_plot = f2.add_subplot(133)
colors = ['red', 'blue', 'green', 'orange', 'pink']
for i in range(d):
    kwargs = {
        'label': f'{i}th derivative error',
        'linewidth': 2.0,
        'color': colors[i]
    }
    yhat_plot.plot(sampling_time[N:], np.abs(yhat_poly[i, N:]-y_samples[i, N:]), **kwargs)
    if i < n:
        kwargs = {
            'label': f'{i}th derivative error, traj',
            'linewidth': 2.0,
            'color': colors[i],
            'linestyle': 'dashed'
        }
        yhat_plot.plot(sampling_time[N:], np.abs(yhat_traj[i, N:]-y_samples[i, N:]), **kwargs)
yhat_plot.legend(loc='upper right')
yhat_plot.grid()
yhat_plot.set_title('Output signal estimation errors')

f2.tight_layout()

f3 = plt.figure(figsize=(12, 6))
sample_x_plot = f3.add_subplot(121)
sample_y_plot = f3.add_subplot(122)
for i, trajectory in enumerate(trajectories):
    sample_x_plot.plot(trajectory[0][0, :], trajectory[0][1, :], linewidth=2.0, label=f'sample {i}')
    sample_y_plot.plot(sampling_time[:N], trajectory[1][0, :], linewidth=2.0, label=f'sample {i}')
sample_x_plot.grid()
sample_y_plot.grid()
sample_x_plot.legend(loc='upper right')
sample_y_plot.legend(loc='upper right')
sample_x_plot.set_title('Trajectory Samples')
sample_x_plot.set_xlabel('x[0]')
sample_x_plot.set_ylabel('x[1]')
sample_y_plot.set_xlabel('t')
sample_y_plot.set_ylabel('y(t)')
sample_y_plot.set_title('Output samples')

plt.show()
