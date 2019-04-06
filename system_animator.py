import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

import copy
"""
About this file:
Runs a simulation of various 2-d systems of ordinary differential equations
"""

class Point(): 
    """Point on the graph, incremented using a system of differential
    equations passed in"""

    def __init__(self, init_position):
        self.position = init_position
        self.age = 0

    def increment(self, system, dt):
        def RK4_step(f, pos, dt):
            """Extension of RK4 for pos in R^n (not sure if this actually how to
            do that"""
            k1 = f(pos)
            k2 = f(pos+0.5*k1*dt)
            k3 = f(pos+0.5*k2*dt)
            k4 = f(pos+k3*dt)
            return pos + (dt * (k1 + 2*k2 + 2*k3 + k4) /6)

        self.position = RK4_step(system, self.position, dt)
        self.age += 1
        

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
    R = 28
    B = 8/3

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


def run(time_chunks, time_chunk_size, system, dt, spawn_interval, spawn_batches, three_d):
    """
    Compute 'time_chunks' frames of data incrementing each point
    'time_chunk_size' times per frame according to 'system'.

    New points are spawned of the grid lines every 'spawn_interval', for
    'spawn_batches'
    """
    state = [[]]
    for t in tqdm(range(1, time_chunks)):
        prev_state = copy.deepcopy(state[t-1])
        curr_state = []
 
        for point in prev_state:  
            for i in range(time_chunk_size):
                point.increment(system, dt)

                # break if the point leaves the graph
                if point.position[0] > X or point.position[1] > Y:
                    break

            curr_state.append(point)
                 
        x_values = np.linspace(-(X/2), (X/2), 10)
        y_values = np.linspace(-(Y/2), (Y/2), 10)
        z_values = np.linspace(-(Z/2), (Z/2), 10)
        if (t-1)%spawn_interval == 0 and t < spawn_batches *spawn_interval:
            for i in x_values:
                for j in y_values:
                    if three_d:
                        for k in z_values:
                            pos = np.array([i, j, k])
                            curr_state.append(Point(pos))
                    else:
                        pos = np.array([i, j])
                        curr_state.append(Point(pos))
            
        state.append(curr_state)  
    return state

def get_data(state, three_d=False):
    """Get state to be plotted"""
    global x, y, z
    if not show_trails:
        x, y = [], []
        if three_d: z = [] 
    for point in state:
        x.append(point.position[0])
        y.append(point.position[1])
        if three_d: z.append(point.position[2]) 
    
    if three_d:
        return (x, y, z) 
    else:
        return (x, y)

def update(t):
    """update function for animation"""
    new_x, new_y = get_data(data[t])
    scat.set_data(new_x, new_y)
    return scat,


def update_three_d(t):
    """update function for animation"""
    new_x, new_y, new_z = get_data(data[t], True)
    scat.set_data((new_x, new_y))
    scat.set_3d_properties(new_z)
    return scat,
 
if __name__ == "__main__":
    T = 10
    X, Y, Z = 40, 40, 40

    Nt = T * 50      # factor here should be inv. proporitional to epsilon
    time_chunks = int(T*25)                 # number of frames computed
    time_chunk_size = int(Nt/time_chunks)    # time steps to compute each frame 

    time_dilation = 1/200 # view function on different time scales
    dt = time_dilation * T/time_chunk_size
    


    fig = plt.figure()
    show_trails = False
     
    three_d = True
    data = run(time_chunks, time_chunk_size, strange_attractor, dt, 50, 1, three_d)

    if three_d:
        ax = fig.add_subplot(111, projection='3d')
        
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(-X/2, X/2)
        ax.set_ylim(-Y/2, Y/2)
        ax.set_zlim(-Z/2, Z/2)

        x, y, z = [], [], []
        scat, = ax.plot(x, y, z, ls=" ", marker="o", color="black", alpha=0.5)
        ani = FuncAnimation(fig, update_three_d, frames=time_chunks, interval=10)
    else:
        ax = fig.add_subplot(111, projection='2d')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xlim(-X/2, X/2)
        plt.ylim(-Y/2, Y/2)
        x, y = [], []
        scat, = ax.plot(x, y, marker="o", ls="", color="black", alpha=0.25)
        ani = FuncAnimation(fig, update, frames=time_chunks, interval=10)

    plt.show()

