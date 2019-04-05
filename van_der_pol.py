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


def run(time_chunks, time_chunk_size, system, dt, spawn_interval, spawn_batches):
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
                 
        if (t-1)%spawn_interval == 0 and t < spawn_batches *spawn_interval:
            for i in range(-int(X/2), int(X/2)):
                for j in range(-int(Y/2), int(Y/2)):
                    pos = np.array([i, j])
                    curr_state.append(Point(pos))
            
        state.append(curr_state)  
    return state

def get_data(state):
    """Get state to be plotted"""
    global x, y
    if not show_trails:
        x, y = [], []
    for point in state:
        x.append(point.position[0])
        y.append(point.position[1])
    return x, y

def update(t):
    """Update function for animation"""
    new_x, new_y = get_data(data[t])
    scat.set_data(new_x, new_y)
    return scat,
 
if __name__ == "__main__":
    T = 25
    X, Y = 20, 10

    Nt = T * 50      # factor here should be inv. proporitional to epsilon
    time_chunks = int(T*25)                 # number of frames computed
    time_chunk_size = int(Nt/time_chunks)    # time steps to compute each frame 

    time_dilation = 1/500 # view function on different time scales
    dt = time_dilation * T/time_chunk_size
    


    fig, ax = plt.subplots()
    plt.title("Van Der Pol")    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(-X/2, X/2)
    plt.ylim(-Y/2, Y/2)
    show_trails = True
     
    data = run(time_chunks, time_chunk_size, pendulum, dt, 50, 1)
    x, y = [], []
    scat, = ax.plot(x, y, marker="o", ls="", color="black", alpha=0.25)
    ani = FuncAnimation(fig, update, frames=time_chunks, interval=10)
    plt.show()

