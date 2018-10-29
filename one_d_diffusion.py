import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

L = 1
Nx = 100

T = 60
dt = 0.001

gamma = 0.05

u = np.zeros(Nx+1)
u_1 = np.zeros(Nx+1)
x = np.linspace(0, L, Nx+1)
dx = x[1]-x[0]

def initial_condition(x):
    if x < L/2:
        return x**2
    else:
        return -x*x
    return np.sin(np.pi * x) + 0.1 * np.sin(10*np.pi*x)

## Forward Euler
def forward_euler(F):
    global u, u_1
    for i in range(0, Nx+1):
        u_1[i] = initial_condition(x[i])

    fig = plt.figure()
    line, = plt.plot(x, u_1)

    def update(t):
        global u_1
        for i in range(1, Nx):
            u[i] = u_1[i] + F*(u_1[i+1] - 2*u_1[i] + u_1[i-1])
        
        u[0], u[Nx] = 0, 0

        u_1 = u
        line.set_data(x, u)
        return line,

    anim = FuncAnimation(fig, update, frames=1000, interval=1, blit=True)
    plt.show()

forward_euler((dt*gamma)/np.power(dx, 2))

## Backward Euler
# reset u and u_1
u = np.zeros(Nx+1)
u_1 = np.zeros(Nx+1)


def backwards_euler(F):
    global u, u_1
    A = np.zeros((Nx+1, Nx+1))
    b = np.zeros(Nx+1)

    for i in range(0, Nx+1):
        u_1[i] = initial_condition(x[i])

    for i in range(1, Nx):
        A[i, i-1] = -F
        A[i, i+1] = -F
        A[i, i] = 1 + 2*F
    A[0,1] = A[Nx, Nx-1] = 0
    A[0,0] = A[Nx, Nx] = 1

    fig = plt.figure()
    line, = plt.plot(x, u_1)
    def update(t):
        global u_1
        b = u_1
        b[0] = b[Nx] = 0
        u[:] = scipy.linalg.solve(A, b)
        u_1 = u
        line.set_data(x, u)
        if t%50 == 0: print(u[5])
        return line,

    anim = FuncAnimation(fig, update, frames=1000, interval=1, blit=True)
    plt.show()

backwards_euler((dt*gamma)/np.power(dx, 2))