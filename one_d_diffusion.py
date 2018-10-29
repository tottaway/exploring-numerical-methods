import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

"""
Most functions in this file were taken or inspired by those found on
http://hplgit.github.io/num-methods-for-PDEs/doc/pub/diffu/sphinx/index.html

About this file:
Runs two simulations of one-dimensional heat diffusion and plots an animation of
heat v. displacement along a rod as it changes over time. The first animation used
an explicit/forward Euler scheme to approximate the state while the second uses
an implicit/backwards Euler scheme.

Unresolved Issues:
Simulations using forwards and backwards methods appear to be identical except
for the fact that they run at different speeds

Things to explore:
* Change initial condition function
    * Discontiuous intitial conditions
    * Sinusoidal function
    * Functions with lots of local noise such as np.sin(np.pi*x)+0.1*np.sin(10*np.pi*x)
* Change properties of Bar (L, gamma)
* Change number of sampled points (Nx, dt)
* Change playback rate
"""
# time inbetween frames (limited by computation speed)
playback_interval = 1

# physical properties
L = 1 # length
gamma = 0.05 # constant describing rate of diffusion (higher is faster)

# Initialize variables
Nx = 100 # total number of differential pieces
u = np.zeros(Nx+1)
u_1 = np.zeros(Nx+1)
x = np.linspace(0, L, Nx+1)

# differential step sizes
dt = 0.001
dx = x[1]-x[0]

# F dictate the size of the step taken on each iteration
F = (dt*gamma)/(dx**2)
def initial_condition(x):
    """Function which returns the initial heat at a given position x"""
    if x < L/2:
        return x**2
    else:
        return -(x**x)

## Forward Euler
def forward_euler():
    # get variables from top-level
    global u, u_1

    # set up initial condition
    for i in range(0, Nx+1):
        u_1[i] = initial_condition(x[i])

    # initialize plot
    fig = plt.figure()
    line, = plt.plot(x, u_1)

    def update(t):
        """Updates plot using Forward Euler"""
        # get state of previous frame
        global u_1

        # apply Euler's at each point
        for i in range(1, Nx):
            # du_dx encodes how heat is changing in an area near the ith element
            du_dx = u_1[i+1] - 2*u_1[i] + u_1[i-1] 
            # delta is the step which is in the direction of du_dx and of size F
            # F encodes information about the values dt, dx, and gamma
            delta = F*du_dx
            u[i] = u_1[i] + delta
        
        # Enforce boundary conditions
        u[0], u[Nx] = 0, 0

        # increment steps
        u_1 = u

        # plot new data
        line.set_data(x, u)
        return line,

    anim = FuncAnimation(fig, update, frames=1000, interval=playback_interval, blit=True)
    plt.show()

forward_euler()

# reset u and u_1
u = np.zeros(Nx+1)
u_1 = np.zeros(Nx+1)


def backwards_euler():
    # get variables from top-level
    global u, u_1

    # initialize variables from algorithm
    A = np.zeros((Nx+1, Nx+1))

    # set up initial conditions
    for i in range(0, Nx+1):
        u_1[i] = initial_condition(x[i])

    # set up A
    for i in range(1, Nx):
        A[i, i-1] = -F
        A[i, i+1] = -F
        A[i, i] = 1 + 2*F
    A[0,1] = A[Nx, Nx-1] = 0
    A[0,0] = A[Nx, Nx] = 1

    # initialize plot
    fig = plt.figure()
    line, = plt.plot(x, u_1)
    
    def update(t):
        """Updates plot using Forward Euler"""
        # get state of previous fram
        global u_1
        
        # enforce boundary conditions
        u_1[0] = u_1[Nx] = 0

        # solve system
        u[:] = scipy.linalg.solve(A, u_1)
        
        # increment steps
        u_1 = u

        # plot new data
        line.set_data(x, u)
        return line,

    anim = FuncAnimation(fig, update, frames=1000, interval=playback_interval, blit=True)
    plt.show()

backwards_euler()