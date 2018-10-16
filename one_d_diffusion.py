import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def initial_condition(x):
    return np.sin(2*np.pi * x)

# Forward Euler
Nx = 1000
L = 1
F = .99

x = np.linspace(0, L, Nx+1)
dx = x[1]-x[0]
u = np.zeros(Nx+1)
u_1 = np.zeros(Nx+1)

for i in range(0, Nx+1):
    u_1[i] = initial_condition(x[i])

fig = plt.figure()
line, = plt.plot(x, u_1)

def update(n):
    global u_1, u
    for i in range(1, Nx):
        u[i] = u_1[i] + F*(u_1[i+1] - 2*u_1[i] + u_1[i-1])
    
    u[0], u[Nx] = 0, 0

    u_1 = u[:]
    line.set_data(x, u)
    if n%50 == 0: print(u[50])
    return line,

anim = FuncAnimation(fig, update, frames=1000, interval=1, blit=True)
plt.show()