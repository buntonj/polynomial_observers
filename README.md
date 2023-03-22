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

Some initial testing is in the `testing.py` and `polyfit_testing.py` files.
