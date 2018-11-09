"""
This file was inspired by this paper:  http://www.math.mcgill.ca/gantumur/docs/reps/RyanSicilianoHH.pdf
some stylistic features where taken from https://www.youtube.com/watch?v=IOkwWYaZbck

About this File:
This file contains definitions for functions which approximate
the solution to a differential using various different methods.

The results of these approximations are then compared with an
exact solution. Next, the accuracy of the different methods, and
how this accuracy depends on step size is analyzed

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
    # return -5*x
    # return 2*x*t
    # return 6 * t**5
    # return 2*np.exp(-5*t) - 4*x
    return -np.sin(t)

def x_t(t):
    # return t
    # return np.exp(t)
    # return np.exp(-5*t)
    # return np.exp(t**2)
    # return t**6
    # return np.exp(-5*t) * (4*np.exp(t) - 2)
    return np.cos(t)

x0 = x_t(0)
T = 6
dt = 0.3
Nt = math.floor(T/dt)

def forward_euler(f, Nt, x0):
    x = x0
    X = []
    time = np.linspace(0, T, Nt)

    h = time[1]

    for t in time:
        X.append(x)
        
        delta_x = h * f(x, t)
        x += delta_x

    
    return X

def trapezoid_method(f, Nt, x0):
    x = x0
    X = []
    time = np.linspace(0, T, Nt)

    h = time[1]

    for t in time:
        X.append(x)

        end_height = x + (h)*f(x, t)
        delta_x = h/2 * (f(end_height, t+h) + f(x, t))
        x += delta_x

    
    return X


def heun(f, Nt, x0):
    x = x0
    X = []
    time = np.linspace(0, T, Nt)

    h = time[1]

    for t in time:
        X.append(x)
        
        midpoint_height = x + (h/2)*f(x, t)
        delta_x = h * f(midpoint_height, t+(0.5*h))
        x += delta_x
    
    return X
    
def rk4(f, Nt, x0):
    def RK4_step(x, t):
        k1 = f(x,t)
        k2 = f(x+0.5*k1*h, t+0.5*h)
        k3 = f(x+0.5*k2*h, t+0.5*h)
        k4 = f(x+k3*h, t+h)
        return h * (k1 + 2*k2 + 2*k3 + k4) /6

    # initial state
    x = x0
    X = []
    time = np.linspace(0, T, Nt)

    h = time[1]

    for t in time:
        X.append(x)

        delta_x = RK4_step(x, t) 
        x += delta_x

    return X

#------------------Plot approximations------------------

# Figure to show approximations
plt.figure()
t = np.linspace(0, T, Nt)

x_euler = forward_euler(dx_dt, Nt, x0)
x_trap = trapezoid_method(dx_dt, Nt, x0)
x_heun = rk4(dx_dt, Nt, x0)
x_rk4 = rk4(dx_dt, Nt, x0)
x_exact = x_t(t)

# Plot comparison
plt.title("Comparison")
plt.plot(t, x_euler, label="Euler", color="blue")
plt.plot(t, x_rk4, label="RK4", color="red")
plt.plot(t, x_trap, label="Trapezoid", color="green")
plt.plot(t, x_heun, label="Heun", color="orange")
plt.plot(t, x_exact, label="exact", color="black")
plt.legend()
plt.show()

#------------------Error Analysis------------------
plt.figure()
plt.title("Error")

Nh = 30
h = np.logspace(-3, -1, Nh)

euler_error = np.zeros(Nh)
trap_error = np.zeros(Nh)
heun_error = np.zeros(Nh)
rk4_error = np.zeros(Nh)
for i, dt in enumerate(h):
    Nt = math.floor(T/dt)
    t = np.linspace(0, T, Nt)
    exact = x_t(t)
    euler = forward_euler(dx_dt, Nt, x0)
    trap = trapezoid_method(dx_dt, Nt, x0)
    heun_res = heun(dx_dt, Nt, x0)
    rk = rk4(dx_dt, Nt, x0)

    euler_error[i] = np.max(np.abs(euler-exact))
    trap_error[i] = np.max(np.abs(trap-exact))
    heun_error[i] = np.max(np.abs(heun_res-exact))
    rk4_error[i] = np.max(np.abs(rk-exact))

# calculate log values
euler_error_log = np.log10(euler_error)
trap_error_log = np.log10(trap_error)
heun_error_log = np.log10(heun_error)
rk4_error_log = np.log10(rk4_error)
h_log = np.log10(h)

euler_slope, euler_intercept, euler_r_value, _, _ = stats.linregress(h_log, euler_error_log)
trap_slope, trap_intercept, trap_r_value, _, _ = stats.linregress(h_log, trap_error_log)
heun_slope, heun_intercept, heun_r_value, _, _ = stats.linregress(h_log, heun_error_log)
rk4_slope, rk4_intercept, rk4_r_value, _, _ = stats.linregress(h_log, rk4_error_log)

plt.plot(h_log, euler_error_log, color="blue", label=("Euler: Order=" + str(np.round(euler_slope, 2))))
plt.plot(h_log, trap_error_log, color="green", label=("Trapezoid: Order=" + str(np.round(trap_slope, 2))))
plt.plot(h_log, heun_error_log, color="orange", label=("Heun: Order=" + str(np.round(heun_slope, 2))))
plt.plot(h_log, rk4_error_log, color="red", label=("RK4:   Order=" + str(np.round(rk4_slope, 2))))
plt.ylabel("log(error)")
plt.xlabel("log(h)")
plt.legend()
plt.show()