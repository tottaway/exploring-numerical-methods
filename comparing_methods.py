"""
This file was inspired by this paper:  http://www.math.mcgill.ca/gantumur/docs/reps/RyanSicilianoHH.pdf
"""
import numpy as np
import matplotlib.pyplot as plt
import math

#------------------Function definitions------------------
def dx_dt(x, t):
    """Derivative of x with respect to time"""
    return np.cos(2*np.pi*t)

def x_t(t):
    return (1/(2*np.pi)) * np.sin(2*np.pi*t)

def forward_euler(f, h, Nt):
    x_1 = 0
    x = np.zeros(Nt)
    x[0] = x_1

    for t in range(1, Nt):
        midpoint_height = x_1 + (h/2)*f(x_1, t*h)
        delta_x = h*f(midpoint_height, t*h)
        x[t] = x_1 + delta_x
        x_1 = x[t]
    
    return x
    
def rk4(f, h, Nt):
    def k1(x, t):
        return h*f(x, t)

    def k2(x, t):
        return h*f(x+0.5*k1(x, t), t+0.5*h)

    def k3(x, t):
        return h*f(x+0.5*k2(x, t), t+0.5*h)
    
    def k4(x, t):
        return h*f(x+k3(x, t), t+h)

    x_1 = 0
    x = np.zeros(Nt)
    x[0] = x_1
    for t in range(1, Nt):
        delta_x = 1/6 * (k1(x_1, t*h) + 2*k2(x_1, t*h) + 2*k3(x_1, t*h) + k4(x_1, t*h))
        x[t] = x_1 + delta_x
        x_1 = x[t]
    
    return x

#------------------Plot approximations------------------
T = 1
dt = 0.01
Nt = math.floor(T/dt)

# Figure to show approximations
plt.figure()
# plt.subplots_adjust(hspace=0.5)
t = np.linspace(0, T, Nt)

# Plot euler approximation
x_euler = forward_euler(dx_dt, dt, Nt)
# plt.subplot(221)
# plt.title("Forward euler's method")
plt.plot(t, x_euler, color="blue")

# plot rk4 approximation
x_rk4 = rk4(dx_dt, dt, Nt)
# plt.subplot(222)
# plt.title("Fourth-order Runge-Kutta")
plt.plot(t, x_rk4, color="red")

# plot rk4 approximation
x_exact = x_t(t)
# plt.subplot(223)
# plt.title("Exact Solution")
plt.plot(t, x_exact, color="orange")
plt.show()

#------------------Error Analysis------------------
plt.figure()
plt.title("Error")

Nh = 1000
h = np.logspace(-3, -1, Nh)

euler_error = np.zeros(Nh)
rk4_error = np.zeros(Nh)
for i, dt in enumerate(h):
    Nt = math.floor(T/dt)
    t = np.linspace(0, T, Nt)
    exact = x_t(t)
    euler = forward_euler(dx_dt, dt, Nt)
    rk = rk4(dx_dt, dt, Nt)

    euler_error[i] = np.mean(np.abs(euler-exact))
    rk4_error[i] = np.mean(np.abs(rk-exact))
plt.loglog(h, euler_error, color="blue")
plt.loglog(h, rk4_error, color="red")
plt.show()