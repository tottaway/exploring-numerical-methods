"""
This file was inspired by this paper:  http://www.math.mcgill.ca/gantumur/docs/reps/RyanSicilianoHH.pdf
some stylistic features where taken from https://www.youtube.com/watch?v=IOkwWYaZbck

About this File:
This file contains definitions for functions which approximate
the solution to a differential using various different methods
(currently only two).

The results of these approximations are then compared with an
exact solution. Next, the accuracy of the different methods, and
how this accuracy depends on step size is analyzed

Current Issues:
* RK4 still performing slightly worse that O(h)

Next Step:
* Implement ADAMS-BASHFORTH-MOULTON PREDICTOR-CORRECTOR METHOD
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import math

#------------------Function and constant definitions------------------
def dx_dt(x, t):
    """Derivative of x with respect to time"""
    # return 1
    # return x
    # return 3 * t**2
    # return 2*np.exp(-5*t) - 4*x
    return np.cos(2*np.pi*t)

def x_t(t):
    # return t
    # return np.exp(t)
    # return t**3
    # return -2*np.exp(-5*t) + 3*np.exp(-4*t)
    return (1/(2*np.pi)) * np.sin(2*np.pi*t)

x0 = 0
T = 1
dt = 0.01
Nt = math.floor(T/dt)

def forward_euler(f, h, Nt, x0):
    x = x0
    X = []
    time = np.linspace(0, T, Nt)

    for t in time:
        # midpoint_height = x_1 + (h/2)*f(x_1, t)
        # delta_x = h * f(midpoint_height, t+(0.5*h))

        # end_height = x_1 + (h)*f(x_1, t)
        # delta_x = h/2 * (f(end_height, t) + f(x_1, t))

        delta_x = h * f(x, t)
        x += delta_x

        X.append(x + delta_x)
    
    return X
    
def rk4(f, h, Nt, x0):
    def RK4_step(y, t, dt):
        k1 = f(y,t)
        k2 = f(y+0.5*k1*dt, t+0.5*dt)
        k3 = f(y+0.5*k2*dt, t+0.5*dt)
        k4 = f(y+k3*dt, t+dt)

        #return dt * G(y,t)
        return dt * (k1 + 2*k2 + 2*k3 + k4) /6

    x0 = 0.0

    # initial state
    x = x0
    time = np.linspace(0, T, Nt)
    X = []

    # time-stepping solution
    for t in time:
        x = x + RK4_step(x, t, h) 
        X.append(x)

    return X

#------------------Plot approximations------------------

# Figure to show approximations
plt.figure()
plt.subplots_adjust(hspace=0.5)
t = np.linspace(0, T, Nt)

# Plot euler approximation
x_euler = forward_euler(dx_dt, dt, Nt, x0)
plt.subplot(221)
plt.title("Forward euler's method")
plt.plot(t, x_euler, color="blue")

# plot rk4 approximation
x_rk4 = rk4(dx_dt, dt, Nt, x0)
plt.subplot(222)
plt.title("Fourth-order Runge-Kutta")
plt.plot(t, x_rk4, color="red")

# plot rk4 approximation
x_exact = x_t(t)
plt.subplot(223)
plt.title("Exact Solution")
plt.plot(t, x_exact, color="orange")

# Plot comparison
plt.subplot(224)
plt.title("Comparison")
plt.plot(t, x_euler, label="Euler", color="blue")
plt.plot(t, x_rk4, label="RK4", color="red")
plt.plot(t, x_exact, label="exact", color="orange")
plt.legend()
plt.show()

#------------------Error Analysis------------------
plt.figure()
plt.title("Error")

Nh = 100
h = np.logspace(-3, -1, Nh)

euler_error = np.zeros(Nh)
rk4_error = np.zeros(Nh)
for i, dt in enumerate(h):
    Nt = math.floor(T/dt)
    t = np.linspace(0, T, Nt)
    dt = t[1]-t[0]
    exact = x_t(t)
    euler = forward_euler(dx_dt, dt, Nt, x0)
    rk = rk4(dx_dt, dt, Nt, x0)

    euler_error[i] = np.mean(np.abs(euler-exact))
    rk4_error[i] = np.mean(np.abs(rk-exact))

# calculate log values
euler_error_log = np.log10(euler_error)
rk4_error_log = np.log10(rk4_error)
h_log = np.log10(h)

euler_slope, euler_intercept, euler_r_value, _, _ = stats.linregress(h_log, euler_error_log)
rk4_slope, rk4_intercept, rk4_r_value, _, _ = stats.linregress(h_log, rk4_error_log)

plt.plot(h_log, euler_error_log, color="blue", label=("Euler: Order=" + str(euler_slope)))
plt.plot(h_log, rk4_error_log, color="red", label=("RK4:   Order=" + str(rk4_slope)))
plt.ylabel("log(error)")
plt.xlabel("log(h)")
plt.legend()
plt.show()