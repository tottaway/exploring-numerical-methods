"""
This file was inspired by this paper:  http://www.math.mcgill.ca/gantumur/docs/reps/RyanSicilianoHH.pdf

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

#------------------Function definitions------------------
def dx_dt(x, t):
    """Derivative of x with respect to time"""
    # return 1
    # return x
    # return 3 * t**2
    return 2*np.exp(-5*t) - 4*x
    # return np.cos(2*np.pi*t)

def x_t(t):
    # return t
    # return np.exp(t)
    # return t**3
    return -2*np.exp(-5*t) + 3*np.exp(-4*t)
    # return (1/(2*np.pi)) * np.sin(2*np.pi*t)

x0 = 1

def forward_euler(f, h, Nt, x0):
    x_1 = x0
    x = np.zeros(Nt)
    x[0] = x_1

    for t in range(1, Nt):
        # midpoint_height = x_1 + (h/2)*f(x_1, t*h)
        # delta_x = h * f(midpoint_height, (t+0.5)*h)

        # end_height = x_1 + (h)*f(x_1, t*h)
        # delta_x = h/2 * (f(end_height, t*h) + f(x_1, t*h))

        delta_x = h * f(x_1, t*h)

        x[t] = x_1 + delta_x
        x_1 = x[t]
    
    return x
    
def rk4(f, h, Nt, x0):
    def k1(x_n, t):
        """slope at startpoint"""
        return f(x_n, t)

    def k2(x_n, t, k_1):
        """slope at midpoint (calculated from k1)"""
        return f(x_n+(k_1*h)/2, t+(h/2))

    def k3(x_n, t, k_2):
        """slope at midpoint (claculated using k2)"""
        return f(x_n+(k_2*h)/2, t+(h/2))
    
    def k4(x_n, t, k_3):
        """slope at endpoint (caluculated using k3)"""
        return f(x_n+k_3*h, t+h)

    x_1 = x0
    x = np.zeros(Nt)
    x[0] = x_1
    for t in range(1, Nt):
        x_1 = x[t-1]
        curr_t = t*h
        k_1 = k1(x_1, curr_t)
        k_2 = k2(x_1, curr_t, k_1)
        k_3 = k3(x_1, curr_t, k_2)
        k_4 = k4(x_1, curr_t, k_1)
        delta_x = h/6 * (k_1 + 2*k_2 + 2*k_3 + k_4)
        x[t] = x_1 + delta_x

    return x

#------------------Plot approximations------------------
T = 1
dt = 0.01
Nt = math.floor(T/dt)

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
    Nt = math.ceil(T/dt)
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