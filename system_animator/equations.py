import numpy as np


def polar_ODE(pos):
    mu=-2
    def r_dot(t, r):
        return r * (1-r**2) + mu*r*np.cos(t)

    def theta_dot(t, r):
        return 1

    return np.array([theta_dot(pos[0], pos[1]), r_dot(pos[0], pos[1])])


def temp(pos):
    def x_dot(x, y):
        return y - x**3

    def y_dot(x, y):
        return -x - y**3

    return np.array([x_dot(pos[0], pos[1]), y_dot(pos[0], pos[1])])


def van_der_pol(pos):
    """Van Der Pol oscilator rewritten as a system of two first order
    ODE's"""

    epsilon = 0.1

    def x_dot(x, y):
        return (1 / epsilon) * (y - ((1/3)*(x**3) - x))

    def y_dot(x, y):
        return -epsilon * x

    return np.array([x_dot(pos[0], pos[1]), y_dot(pos[0], pos[1])])


def pendulum(pos):
    """Model of a pendulum"""

    def x_dot(x, y):
        return y

    def y_dot(x, y):
        return -np.sin(x)

    return np.array([x_dot(pos[0], pos[1]), y_dot(pos[0], pos[1])])


def damped_pendulum(pos):
    """Model of a damped pendulum"""

    def x_dot(x, y):
        return y

    def y_dot(x, y):
        return -0.1*y - np.sin(x)

    return np.array([x_dot(pos[0], pos[1]), y_dot(pos[0], pos[1])])


def strange_attractor(pos):
    """Model of a pendulum"""

    P = 10
    R = 350
    B = 8/3
    #P = 10
    #R = 35
    #B = 5

    def x_dot(x, y, z):
        return P * (y - x)

    def y_dot(x, y, z):
        return R*x - y - x*z

    def z_dot(x, y, z):
        return x*y - B*z

    return np.array([x_dot(pos[0], pos[1], pos[2]),
                     y_dot(pos[0], pos[1], pos[2]),
                     z_dot(pos[0], pos[1], pos[2])])

    
def rossler_attractor(pos):
    """Model of a pendulum"""

    A = 0.2
    B = 0.2
    C = 5.7

    def x_dot(x, y, z):
        return -(y + z)

    def y_dot(x, y, z):
        return x + A*y

    def z_dot(x, y, z):
        return B + x*z - C*z

    return np.array([x_dot(pos[0], pos[1], pos[2]),
                     y_dot(pos[0], pos[1], pos[2]),
                     z_dot(pos[0], pos[1], pos[2])])


equations = {
    "polar_ODE": polar_ODE,
    "temp": temp,
    "van_der_pol": van_der_pol,
    "pendulum": pendulum,
    "damped_pendulum": damped_pendulum,
    "strange_attractor": strange_attractor,
    "rossler_attractor": rossler_attractor
}
