import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from nonlinear_system.sample_odes import AckermanModel
from nonlinear_system.ct_system import ContinuousTimeSystem
from matplotlib.patches import Rectangle, Polygon, Arrow, Wedge
from matplotlib.collections import PatchCollection
from matplotlib.animation import FuncAnimation, PillowWriter

np.random.seed(0)


def control_input(t, y, x=None) -> np.ndarray:
    # if the system has control inputs, we can calculate them here with time-varying output or state feedback
    f = 2.0
    mag = 0.5
    u = np.zeros((2, 3))
    # acceleration and its derivatives
    u[0, 0] = 0.1
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
num_steps = 200

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
# gs = mpl.gridspec.GridSpec(6, 6, figure=vfig)
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
car_zero = vax.scatter(0.0, 0.0, s=20, color='black', marker='x', zorder=10)
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
               'color': 'red',
               'width': 0.05}
heading_bx = 0.25*car_len
heading_by = 0.0
heading_len = 0.75*car_len
heading_vec = Arrow(heading_bx, heading_by, heading_len, 0.0, **arrow_props)

bound_props = {'fill': True,
               'color': 'green',
               'alpha': 0.3,
               'width': 0.75*heading_len,
               'zorder': -1}
heading_bound = Wedge((heading_bx, heading_by), heading_len, np.degrees(-np.pi/8), np.degrees(np.pi/8), **bound_props)

pos_props = {'fill': True,
             'color': 'green',
             'alpha': 0.2,
             'zorder': -1}
unc_x = 0.25
unc_y = 0.25
est_x = x[0, t] + 0.01
est_y = x[1, t] - 0.01
pos_x = est_x - unc_x
pos_y = est_y - unc_y
pos_bound = Rectangle((pos_x, pos_y), unc_x*2, unc_y*2, **pos_props)
est_zero = vax.scatter(est_x, est_y, s=20, color='red', marker='x', zorder=15)
# car_body = Rectangle(caranchor, car_len, car_width, **car_props)

# car_base = PatchCollection([car_body, lr_wheel, rr_wheel], match_original=True)

trans = mpl.transforms.Affine2D().rotate(x[2, 0]) + mpl.transforms.Affine2D().translate(x[0, 0], x[1, 0]) + vax.transData
# t_end = vax.transData + trans

car_base = PatchCollection([car_body, lr_wheel, rr_wheel], match_original=True)

heading_vec.set_transform(trans)
heading_bound.set_transform(trans)
car_base.set_transform(trans)
lf_wheel.set_transform(trans)
rf_wheel.set_transform(trans)

vax.add_collection(car_base)
vax.add_patch(pos_bound)
vax.add_patch(lf_wheel)
vax.add_patch(rf_wheel)
vax.add_patch(heading_bound)
vax.add_patch(heading_vec)
vax.set_aspect('equal')

steer_widget_width = 0.15
steer_widget_height = 0.15
# stax = vfig.add_subplot(gs[:-1, -1], projection='polar')
stax = vfig.add_axes((0.17, 0.2, steer_widget_width, steer_widget_height), projection='polar')
steer_rad = 0.5
true_phi_line, = stax.plot([x[4, t], x[4, t]], [0., steer_rad], linewidth=1, c='black')
unc_phi = np.pi/6.
est_phi = x[4, t] + 0.001
est_bounds = [est_phi - unc_phi, est_phi + unc_phi]
est_phi_line, = stax.plot([est_phi, est_phi], [0, steer_rad], linewidth=1, c='red')
steer_bound = Rectangle((est_phi-unc_phi, 0.0), unc_phi, 2*steer_rad, color='green', alpha=0.3, zorder=-1)
stax.add_patch(steer_bound)
stax.set_aspect('equal')
stax.set_theta_zero_location("N")
stax.set_rticks([10.0])
stax.set_rlim(0.0, 1.25*steer_rad)
theta_lab = [f'{i}' for i in range(0, 180, 45)] + [f'{i}' for i in range(-180, 0, 45)]
thetas = list(range(0, 360, 45))
stax.set_thetagrids(thetas, theta_lab)
stax.set_rmax(steer_rad)

# velax = vfig.add_subplot(gs[-1, -1])
velax = vfig.add_axes((0.17, 0.15, steer_widget_width, steer_widget_height*0.1))
est_vel = x[3, t] - 0.01
unc_vel = 0.1
bar_ht = 1.0
ht = 0.5*bar_ht
vmax = 4.0
true_vel_line, = velax.plot([x[3, t], x[3, t]], [0.0, bar_ht], color='black',lw=2)
est_vel_line, = velax.plot([est_vel, est_vel], [0.0, bar_ht], color='red', lw=2)
vel_bound = Rectangle((est_vel-unc_vel, 0.0), 2*unc_vel, bar_ht, alpha=0.3, color='green', zorder=-1)
velax.add_patch(vel_bound)
velax.set_xlim(0.0, vmax)
velax.set_ylim(0.0, bar_ht)
velax.grid()
velax.yaxis.set_visible(False)
# vfig.subplots_adjust(wspace=0, hspace=0)


def update(i):
    left, right = compute_lr_angles(x[4, i])
    lf_wheel.set_angle(left)
    rf_wheel.set_angle(right)
    trans = mpl.transforms.Affine2D().rotate(x[2, i]) + mpl.transforms.Affine2D().translate(x[0, i], x[1, i]) + vax.transData
    car_zero.set_offsets([x[0, i], x[1, i]])
    est_x = x[0, i] - 0.01
    est_y = x[1, i] + 0.01
    est_zero.set_offsets([est_x, est_y])
    car_base.set_transform(trans)
    lf_wheel.set_transform(trans)
    rf_wheel.set_transform(trans)
    heading_vec.set_transform(trans)
    heading_bound.set_transform(trans)
    heading_bound.set_theta1(np.degrees(-np.pi/8))
    heading_bound.set_theta2(np.degrees(np.pi/8))
    pos_bound.set_xy((x[0, i]-unc_x, x[1, i]-unc_y))
    pos_bound.set_width(2*unc_x)
    pos_bound.set_height(2*unc_y)
    vax.set_title(f'Frame {i}')
    true_phi_line.set_xdata([x[4, i], x[4, i]])
    est_phi = x[4, i] + 0.001
    est_phi_line.set_xdata([est_phi, est_phi])
    steer_bound.set_xy((est_phi-unc_phi, 0.0))
    steer_bound.set_width(2*unc_phi)
    true_vel_line.set_xdata([x[3, i], x[3, i]])
    est_vel = x[3, i] - 0.01
    est_vel_line.set_xdata([est_vel, est_vel])
    vel_bound.set_xy((est_vel-unc_vel, 0.0))
    vel_bound.set_width(2*unc_vel)
    return car_body, lf_wheel, rf_wheel, heading_vec, heading_bound, pos_bound, true_phi_line, est_phi_line, steer_bound


# Now we need to compute the transformations to apply to each part

anim = FuncAnimation(vfig, update, frames=np.arange(0, num_steps), repeat=True, interval=dt*100, blit=False)
anim.save('car_test.gif', dpi=300, writer=PillowWriter(fps=int(1./dt)))
plt.show()
