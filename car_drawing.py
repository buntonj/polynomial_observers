import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from nonlinear_system.sample_odes import AckermanModel
from nonlinear_system.ct_system import ContinuousTimeSystem
from matplotlib.patches import Rectangle, Polygon, Arrow
from matplotlib.collections import PatchCollection
from matplotlib.animation import FuncAnimation, PillowWriter

np.random.seed(0)


def control_input(t, y, x=None) -> np.ndarray:
    # if the system has control inputs, we can calculate them here with time-varying output or state feedback
    f = 2.0
    mag = 0.5
    u = np.zeros((2, 3))
    # acceleration and its derivatives
    u[0, 0] = 0.0
    u[0, 1] = 0.0
    u[0, 2] = 0.0

    u[1, 0] = -mag*np.sin(f*t)
    u[1, 1] = -f*mag*np.cos(f*t)
    u[1, 2] = (f**2.0)*mag*np.sin(f*t)
    return u


def compute_lr_angles(phi: float) -> float:
    left = np.degrees(np.arctan2(2.0*axle_sep*np.sin(phi), 2*ODE.axle_sep*np.cos(phi)-np.sign(phi)*ODE.wheel_sep*np.sin(phi)))
    right = np.degrees(np.arctan2(2.0*axle_sep*np.sin(phi), 2*ODE.axle_sep*np.cos(phi)+np.sign(phi)*ODE.wheel_sep*np.sin(phi)))
    return left, right


axle_sep = 0.5
wheel_sep = 0.6*axle_sep
n = 5
num_steps = 50

x = np.empty((n, num_steps))
x0 = 5.0*(np.random.rand(n)-0.5)
x0[2] = np.pi/6.0
x0[4] = np.pi/4.0  # np.clip(x0[4], -np.pi, np.pi)
x[:, 0] = x0
dt = 0.05
ODE = AckermanModel(axle_sep, wheel_sep)
sys = ContinuousTimeSystem(ODE, x0=x0, dt=0.05)

for i in range(1, num_steps):
    u = control_input(i*dt, None)[:, 0]
    x[:, i], _ = sys.step(control_input(i*dt, None)[:, 0])

vfig = plt.figure(figsize=(8, 8))
vax = vfig.add_subplot()
vax.set_xlim((-4.0, 4.0))
vax.set_ylim((-4.0, 4.0))
t = 0

# First, we'll build the car shapes with no transformation applied.

wheel_dep = ODE.wheel_sep/4  # how fat will the wheels be
wheel_diam = wheel_dep*3.0  # how big are they in diameter

wheel_params = {'color': 'black',
                'fill': True,
                'zorder': 1.0}

# RIGHT (PASSENGER) REAR WHEEL
rr_anchor = (0.0 - wheel_diam/2., 0.0 - ODE.wheel_sep/2. - wheel_dep/2.)
rr_wheel = Rectangle(rr_anchor, wheel_diam, wheel_dep, **wheel_params)

# LEFT (DRIVER) REAR WHEEL
lr_anchor = (0.0 - wheel_diam/2., 0.0 + ODE.wheel_sep/2. - wheel_dep/2.)
lr_wheel = Rectangle(lr_anchor, wheel_diam, wheel_dep, **wheel_params)

lf_angle, rf_angle = compute_lr_angles(x[4, t])

# RIGHT (PASSENGER) FRONT WHEEL
rf_anchor = (0.0 + ODE.axle_sep - wheel_diam/2., 0.0 - ODE.wheel_sep/2. - wheel_dep/2.)
rf_wheel = Rectangle(rf_anchor, wheel_diam, wheel_dep, angle=rf_angle, rotation_point='center', **wheel_params)

# LEFT (DRIVER) FRONT WHEEL
lf_anchor = (0.0 + ODE.axle_sep - wheel_diam/2., 0.0 + ODE.wheel_sep/2. - wheel_dep/2.)
lf_wheel = Rectangle(lf_anchor, wheel_diam, wheel_dep, angle=lf_angle, rotation_point='center', **wheel_params)

base_LR_pad = wheel_dep/2.0
base_UD_pad = -wheel_dep/4.0
caranchor = (0.0 - wheel_diam/2.0 - base_LR_pad, 0.0 - ODE.wheel_sep/2.0 - wheel_dep/2.0 - base_UD_pad)
car_len = ODE.axle_sep + wheel_diam + base_LR_pad*2.0
car_width = ODE.wheel_sep + base_UD_pad*2.0 + wheel_dep
car_rr = np.array(caranchor)
car_lr = car_rr + np.array([0.0, car_width])
car_rf = car_rr + np.array([car_len, 0.15*car_width])
car_lf = car_lr + np.array([car_len, -0.15*car_width])
car_pts = np.vstack([car_rr, car_lr, car_lf, car_rf])
car_props = {'fill': True,
             'edgecolor': 'black',
             'capstyle': 'round',
             'zorder': -1}
car_body = Polygon(car_pts, **car_props)

arrow_props = {'fill': True,
               'color': 'red'}
heading_vec = Arrow(caranchor[0]+0.25*car_len, caranchor[1], car_len, 0.0, **arrow_props)
# car_body = Rectangle(caranchor, car_len, car_width, **car_props)

# car_base = PatchCollection([car_body, lr_wheel, rr_wheel], match_original=True)

trans = mpl.transforms.Affine2D().rotate(x[2, 0]) + mpl.transforms.Affine2D().translate(x[0, 0], x[1, 0]) + vax.transData
# t_end = vax.transData + trans

car_base = PatchCollection([car_body, lr_wheel, rr_wheel], match_original=True)
heading_vec.set_transform(trans)
car_base.set_transform(trans)
lf_wheel.set_transform(trans)
rf_wheel.set_transform(trans)
vax.add_collection(car_base)
vax.add_patch(lf_wheel)
vax.add_patch(rf_wheel)
vax.add_patch(heading_vec)
vax.set_aspect('equal')

def update(i):
    left, right = compute_lr_angles(x[4, i])
    lf_wheel.set_angle(left)
    rf_wheel.set_angle(right)
    trans = mpl.transforms.Affine2D().rotate(x[2, i]) + mpl.transforms.Affine2D().translate(x[0, i], x[1, i]) + vax.transData
    car_base.set_transform(trans)
    lf_wheel.set_transform(trans)
    rf_wheel.set_transform(trans)
    heading_vec.set_transform(trans)
    vax.set_title(f'Frame {i}')
    return car_body, lf_wheel, rf_wheel, heading_vec


# Now we need to compute the transformations to apply to each part

anim = FuncAnimation(vfig, update, frames=np.arange(0, num_steps), repeat=True, interval=dt*100, blit=False)
anim.save('car_test.gif', dpi=300, writer=PillowWriter(fps=5))
plt.show()
