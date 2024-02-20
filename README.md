# polynomial_observers
This repository implements a few tools for proof-of-concept exploration.  The high-level problem is to estimate the state of a dynamical system
$$\dot{x} = f(x,u) \\ y = h(x).$$
We do this by appealing to the definition of _differential observability_, which states that if a system is _differentially observable of order $d$_, then there is an injective function $L$ that produces the current state $x(t)$ from the current value of $y(t)$ and its first $d$ derivatives, i.e.:

$$ L(y(t), \dot{y}(t), \cdots, y^{(d)}(t)) = x(t).$$

This definition is in fact more general than this (it appeals to a more differential geometric viewpoint of the observability problem), but for linear systems it reduces to the standard notions of observability based on the observability Gramian/matrix.

The existence of the injective map $L$ means that estimating the current state $x(t)$ is equivalently to estimating the system output $y(t)$ and $d$ of its derivatives.  However, we only receive information about the output $y(t)$ through its evaluations at sampling instants.
The task, then, is to produce an estimate of the output $y(t)$ and its first $d$ derivatives at any given time $t\in\mathbb{R}_{\geq 0}$ using only samples of $y$ at a handful of discrete points.

This repository implements a generic nonlinear system python class `ContinuousTimeSystem` that uses the `scipy.integrate.solve_ivp` function to numerically integrate the nonlinear differential equations above.
The repository also contains a python class `PolyEstimator` for solving the above problem by fitting finite windows of output function evaluations with a polynomial of degree $d$, then differentiating it at a given time $t$.

The main theoretical contribution is the ability to compute an error bound for this polynomial-based filter in an _online_ fashion.  Typical bounds for observer error are of the input-to-error-state variety, where the observer will (asymptotically) converge to the true state up to some neighborhood, the size of which must be determined offline by (for instance) bounding the magnitude of the expected noise.  In this approach, however, even if the signal sees less noise than this quantity for some time, the convergence guarantee is inflated for _all time_.  A related notion explored in the past is Quasi-ISS or input-to-state-dynamically-stable (ISDS) observers (see, for example [here](http://liberzon.csl.illinois.edu/research/CDC09_0805_MS.pdf), [here](https://ieeexplore.ieee.org/document/1429345), or [here](https://www.sciencedirect.com/science/article/pii/S0167691114002692?fr=RR-2&ref=pdf_download&rr=7d12308868e32a91).  Even these observers, however, need an __estimate of the experienced disturbance/noise__ to function.

Our theory shows that the polynomial fitting residuals (the quantities $y(t) - \hat{y}(t)$ for each of the discrete sample instants $t = t_0, t_1, ... , t_N$) serve as a valid proxy for these disturbance estimates.  As a result, we can use the _online_ polynomial fitting residuals to provide real-time error bounds that grow and shrink depending on the experienced disturbances, nonlinearities, and noise (and the polynomial fit's natural response to them).

In a single picture, the error bounds we get look more like this, for a basic bicycle/Ackerman steering model:

![ackerman_example](https://github.com/buntonj/polynomial_observers/blob/main/car_test.mp4)

Here we are estimating the vehicle's position ($x$ and $y$ coordinates), orientation (angle from horizontal, $\theta$), linear velocity magnitude, and steering angle (from straight) from noisy "GPS-style" measurements of just $x-y$ coordinates.  From this output, the steering angle requires computing two derivatives, and is thus the most impacted by noise, but is easier to estimate when the car has high centripetal acceleration (which we see in the error bounds).

Some initial testing is in the `testing.py`, `polyfit_testing.py`, `bound_testing.py` and `ackerman_testing.py` files.
