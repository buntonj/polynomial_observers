from nonlinear_system.ct_system import ContinuousTimeSystem
from nonlinear_system.sample_odes import LTI
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import scipy
import cvxpy as cvx

np.random.seed(0)
verbose = False
##############################################################
#                     TIME  PARAMETERS                       #
##############################################################
integration_dt = 0.1
num_integration_steps = 500

##############################################################
#                    SYSTEM PARAMETERS                       #
##############################################################
noise_mag = 0.0  # 000001  # magnitude of noise to be applied to outputs

n = 2  # system state dimension
m = 1  # control input dimension
p = 1  # output dimension
A = np.array([[0., 1.],
              [0., 0.]])
B = np.array([[0.],
              [1.]])
C = np.array([[1., 0.]])
ODE = LTI(A, C, B=B)
uderivs = ODE.uderivs  # number of control input derivatives we need to provide

# lazily placing poles at same location
evals = -10.0
l1 = -2.*evals
l2 = evals**2.0
observer_L = np.array([[l1],
                       [l2]])
observer_ODE = LTI(A, C, B=np.eye(n))

dt_A = scipy.linalg.expm((A-observer_L@C)*integration_dt)
print(dt_A)
N = 20


def observer_control(t, u, y, yhat):
    return observer_L @ (y - yhat)  # + B @ u


def noise_generator(t: float, mag: float, p: int) -> np.ndarray:
    return mag*(np.random.rand(p)-0.5)


def control_input(t, y, x=None) -> np.ndarray:
    # if the system has control inputs, we can calculate them here with time-varying output or state feedback
    f = 2.0
    mag = 1.0
    u = np.zeros((1,))
    u[0] = mag*np.cos(f*t)
    # acceleration and its derivatives

    # u[0, 0] = mag*np.cos(f*t)
    # u[0, 1] = -f*mag*np.sin(f*t)
    # u[0, 2] = -(f**2.0)*mag*np.cos(f*t)
    return u


def build_LP(residuals, ebound, ubound, max=True):
    W = residuals.shape[1]  # how many residuals are there?
    U = cvx.Variable(W//2)
    E = cvx.Variable(2)
    constraints = []
    Ap = [np.linalg.matrix_power(dt_A, t) for t in range(W)]
    for t in range(W//2):
        constraint = 0.0
        for s in range(t):
            constraint += C @ Ap[s] @ B * U[t-1-s]
        constraint += C @ Ap[t] @ E

        constraints.append(constraint == residuals[:, t])
        constraints.append(-ubound <= U[t])
        constraints.append(U[t] <= ubound)

    constraints.append(cvx.norm_inf(E) <= ebound)
    # constraints.append(E == 0)
    cost = np.ones(2) @ Ap[W-1] @ E
    for t in range(W//2-1):
        cost += np.ones(2) @ Ap[t] @ B * U[W//2-2-t]

    if max:
        obj = cvx.Maximize(cost)
    else:
        obj = cvx.Minimize(cost)
    prob = cvx.Problem(obj, constraints)
    val = prob.solve()
    print(prob.status)
    return val


# setting up arrays to hold the state, output, and output derivatives at all integration times
x = np.zeros((n, num_integration_steps))
xhat = np.zeros((n, num_integration_steps))
y = np.zeros((p, num_integration_steps))
yhat = np.zeros((p, num_integration_steps))
LP_ub = np.zeros((num_integration_steps-N,))
LP_lb = np.zeros((num_integration_steps-N,))

# setting up arrays for errors and bounds in estimation
residuals = np.empty((p, num_integration_steps))
bounds = np.zeros((n, num_integration_steps))

# setting up arrays for variables at sampling instants (could be done with index splicing)
u = np.empty((m, num_integration_steps-1))
u_observer = np.empty((n, num_integration_steps-1))

# time
integration_time = np.zeros((num_integration_steps,))

# initializing the ODE
x0 = 5.0*(np.random.rand(n)-0.5)
xhat0 = x0  # 5.0*(np.random.rand(n)-0.5)
x[:, 0] = x0
x[:, 0] = xhat0

init_error = np.linalg.norm(x0 - xhat0)

sys = ContinuousTimeSystem(ODE, x0=x0, dt=integration_dt, solver='RK45')
observer = ContinuousTimeSystem(observer_ODE, x0=xhat0, dt=integration_dt, solver='RK45')

y[:, 0] = sys.y
yhat[:, 0] = observer.y

# SIMULATION LOOP
for t in range(1, num_integration_steps):
    # select the control input
    u[:, t-1] = control_input(sys.t, y[:, t-1])
    u_observer[:, t-1] = observer_control(sys.t, u[:, t-1], y[:, t-1], yhat[:, t-1])

    x[:, t], y[:, t] = sys.step(u[:, t-1])  # stepping only requires input value
    xhat[:, t], yhat[:, t] = observer.step(u_observer[:, t-1])

    residuals[:, t] = y[:, t] - yhat[:, t]
    integration_time[t] = sys.t

    if t >= N:
        LP_ub[t-N] = build_LP(residuals[:, t-N:t], ebound=0.1, ubound=1.0, max=True)
        LP_lb[t-N] = build_LP(residuals[:, t-N:t], ebound=0.1, ubound=1.0, max=False)
        print(f'LP SOLUTION: {LP_ub[t-N]-LP_lb[t-N]:.2f}')

    if verbose:
        print(f'Completed timestep {t}, t = {sys.t:.1e}, state = {sys.x}')


# SAVING THE DATA HERE
savedata = {'x': x,
            'xhat': xhat}

path = './tmp/linear'
if not os.path.exists(path):
    os.mkdir(path)
filename = path + '/last_run.p'
with open(filename, 'wb') as f:
    pickle.dump(savedata, f)


gridrows = int(np.ceil(n/4))
gridcols = min(4, n)
size = (5*gridcols, 5)
f, axs = plt.subplots(nrows=gridrows, ncols=gridcols, figsize=size)
for i in range(n):
    ax = axs.ravel()[i]
    ax.plot(integration_time, x[i, :], linewidth=2.0, c='blue', label='truth')
    ax.plot(integration_time, xhat[i, :], linewidth=2.0, c='red', linestyle='dashed', label='observer estimate')
    ax.fill_between(integration_time[N:], xhat[i, N:] + LP_ub, xhat[i, N:]+LP_lb)
    ax.set_xlabel('time (s)')
    ax.set_ylabel(f'x[{i}](t)')
    ax.legend()
    ax.grid()
f.tight_layout()

f2, ax2 = plt.subplots()
ax2.plot(integration_time[N:], LP_lb)
ax2.plot(integration_time[N:], LP_ub)
plt.show()
