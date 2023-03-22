from nonlinear_system.ct_system import ContinuousTimeSystem
from moving_polyfit.moving_ls import PolyEstimator
import numpy as np
import matplotlib.pyplot as plt

c = 0.5
f = 500


def phi(t, x):
    if t <= 2.5:
        return 0.0  # 0.5*c*np.sum(x**2.0)  # phi(x) = x^Tx nonlinearity
    else:
        return t*np.cos(f*t)  # t**3.0 - 2.5  # -3*c*np.sum(x**2.0)


def dphidt(t, x):
    if t <= 2.5:
        return 0.0  # c*np.dot(x, rhs(t, x, None))
    else:
        return -f*t*np.sin(f*t) + np.cos(f*t)  # 3.0*t**2.0  # 1.0  # -3*2*c*np.dot(x, rhs(t, x, None))


def rhs(t, x, u):
    '''
    RHS for n-dimensional integrator ODE
    dx[i]/dt = x[i+1]
    dx[n]/dt = phi(x)
    '''
    rhs = np.zeros_like(x)
    rhs[0:-1] = x[1:]
    rhs[-1] = phi(t, x)
    return rhs


def output_fn(t, x, u):
    return x[0]


def control_input(t, y):
    return 0.0


n = 2  # system state dimension
m = 1  # control input dimension
p = 1  # output dimension
num_steps = 10000

x = np.empty((n, num_steps))
y = np.empty((p, num_steps))
u = np.empty((m, num_steps-1))
time = np.zeros((num_steps,))
np.random.seed(2)
x0 = np.random.uniform(low=-1.0, high=1.0, size=n)  # generate a random initial state
x[:, 0] = x0
dt = 0.0005

sys = ContinuousTimeSystem(2, rhs, h=output_fn, x0=x0, dt=dt)
y[:, 0] = sys.y
print("Initialized CT system object.")

d = 4  # degree of estimation polynomial
N = 1000  # number of samples
estimator = PolyEstimator(d, N, dt)
theta = np.empty((d, num_steps-N))
dphi_hat = np.empty((num_steps-N,))
dphi = np.empty((num_steps-N,))
yhat = np.empty((d, num_steps-N))
ytrue = np.zeros((d, num_steps-N))

for t in range(1, num_steps):
    u[:, t-1] = control_input(sys.t, y[:, t-1])
    x[:, t], y[:, t] = sys.step(u[:, t-1])
    time[t] = sys.t
    if t >= N:
        theta[:, t-N] = estimator.fit(y[0, t-N:t])
        for i in range(d):
            yhat[i, t-N] = estimator.differentiate((N-1)*dt, i)

        for i in range(sys.n):
            ytrue[i, t-N] = x[i, t]
        ytrue[sys.n, t-N] = phi(sys.t, x[:, t])
        ytrue[sys.n+1, t-N] = dphidt(sys.t, x[:, t])
        dphi[t-N] = dphidt(sys.t, x[:, t])

    print(f'Completed timestep {t}, t = {sys.t:.1e}, state = {sys.x}')


f = plt.figure(figsize=(10, 10))
x0_plot = f.add_subplot((221))
x1_plot = f.add_subplot((222))
traj_plot = f.add_subplot((223))
u_plot = f.add_subplot((224))

x0_plot.plot(time, x[0, :], linewidth=2.0, c='blue')
x0_plot.plot(time[N:], yhat[0, :], linewidth=2.0, c='red', linestyle='dashed')
x0_plot.set_xlabel('time (s)')
x0_plot.set_ylabel('x[0]')
x0_plot.grid()

x1_plot.plot(time, x[1, :], linewidth=2.0, c='blue')
x1_plot.plot(time[N:], yhat[1, :], linewidth=2.0, c='red', linestyle='dashed')
x1_plot.set_xlabel('time (s)')
x1_plot.set_ylabel('x[1]')
x1_plot.grid()

traj_plot.plot(x[0, :], x[1, :], linewidth=2.0, c='blue')
traj_plot.plot(yhat[0, :], yhat[1, :], linewidth=2.0, c='red', linestyle='dashed')
traj_plot.scatter(x[0, 0], x[1, 0], s=50, marker='*')
marg = 0.1
traj_plot.set_xlim(x[0, :].min()-marg, x[0, :].max()+marg)
traj_plot.set_ylim(x[1, :].min()-marg, x[1, :].max()+marg)
traj_plot.set_xlabel('x[0]')
traj_plot.set_ylabel('x[1]')
traj_plot.grid()

u_plot.plot(time[1:], u[0, :], linewidth=2.0, c='red')
u_plot.set_xlabel('time (s)')
u_plot.set_ylabel('u')
u_plot.grid()

f.tight_layout()

f2 = plt.figure(figsize=(15, 6))
thetas_plot = f2.add_subplot(131)

for i in range(d):
    thetas_plot.plot(time[N:], theta[i, :], linewidth=2.0, label=f'Theta[{i}]')
thetas_plot.legend()
thetas_plot.grid()

est_plot = f2.add_subplot(132)
est_plot.plot(time[N:], dphi, linewidth=2.0, c='blue', label='truth')
est_plot.plot(time[N:], yhat[3, :], linewidth=2.0, c='red', label='estimate')
est_plot.legend()
est_plot.grid()

yhat_plot = f2.add_subplot(133)
colors = ['red', 'blue', 'green', 'orange', 'pink']
for i in range(d):
    kwargs = {
        'label': f'{i}th derivative',
        'linewidth': 2.0,
        'color': colors[i]
    }
    yhat_plot.plot(time[num_steps//2+2*N:], np.abs(yhat[i, num_steps//2+N:]-ytrue[i, num_steps//2+N:]), **kwargs)
    yhat_plot.plot(time[N:num_steps//2], np.abs(yhat[i, :num_steps//2-N]-ytrue[i, :num_steps//2-N]), **kwargs)
yhat_plot.legend()
yhat_plot.grid()
yhat_plot.set_title('Output signal estimation errors')

f2.tight_layout()

plt.show()
