import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

import copy

import sys
import argparse
from equations import equations
from config import config
""" About this file:
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


def run(time_chunks, time_chunk_size, system, dt, three_d):
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

            # drop the point leaves the graph (TODO: put in logic for polar)
            if polar or not point.position[0] > X or not point.position[1] > Y:
                curr_state.append(point)
                 
        x_values = np.linspace(-(X/2), (X/2), config["x_points"])
        y_values = np.linspace(-(Y/2), (Y/2), config["y_points"])
        z_values = np.linspace(-(Z/2), (Z/2), config["z_points"])
        if (t-1)%config["spawn_interval"] == 0 and t < config["spawn_batches"] * config["spawn_interval"]:
            if three_d:
                for i in x_values:
                    for j in y_values:
                            for k in z_values:
                                pos = np.array([i, j, k])
                                curr_state.append(Point(pos))
            elif polar:
                for i in x_values:
                    for j in y_values:
                        pos = np.array([np.arctan(i/j), np.sqrt(i**2 + j**2)])
                        curr_state.append(Point(pos))
            else:
                for i in x_values:
                    for j in y_values:
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

def get_polar_data(state, three_d=False):
    """Get state to be plotted"""
    global x, y
    if not show_trails:
        x, y = [], []
    for point in state:
        t_i = point.position[0]
        r_i = point.position[1]
        x.append(np.abs(r_i) * np.cos(t_i))
        y.append(np.abs(r_i) * np.sin(t_i))
    
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

def polar_update(t):
    new_x, new_y = get_polar_data(data[t])
    scat.set_data(new_x, new_y)
    return scat,
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configure simulation')
    parser.add_argument('system', type=str, help='system to be simulated')
    parser.add_argument('--3d', dest='three_d', action='store_const',
                    const=True, default=False,
                    help='Runs simulation in 3d')
    parser.add_argument('--show-trails', dest='show_trails', action='store_const',
                    const=True, default=False,
                    help='Stops programs from removing previous point')
    parser.add_argument('--polar', dest='polar', action='store_const',
                    const=True, default=False,
                    help='Indicates function is polar (theta, r)')
    args = parser.parse_args()

    T = config["T"]
    X, Y, Z = config["X"], config["Y"], config["Z"]

    Nt = T * config["Nt"]      # factor here should be inv. proporitional to epsilon
    time_chunks = int(T*25)                 # number of frames computed
    time_chunk_size = int(Nt/time_chunks)    # time steps to compute each frame 

    time_dilation = config["time_dilation"] # view function on different time scales
    dt = time_dilation * T/time_chunk_size
    


    fig = plt.figure()
    show_trails = args.show_trails
     
    three_d = args.three_d
    # displays in polar
    polar = args.polar
    data = run(time_chunks, time_chunk_size, equations[args.system], dt, three_d)

    if three_d:
        ax = fig.add_subplot(111, projection='3d')
        
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(-X/2, X/2)
        ax.set_ylim(-Y/2, Y/2)
        #ax.set_zlim(-Z/2, Z/2)
        ax.set_zlim(0, Z)

        x, y, z = [], [], []
        scat, = ax.plot(x, y, z, ls=" ", marker="o", markersize=5, color="black", alpha=0.5)
        updater = update_three_d
    else:
        ax = fig.add_subplot(111)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xlim(-X/2, X/2)
        plt.ylim(-Y/2, Y/2)

        x, y = [], []
        scat, = ax.plot(x, y, marker="o", markersize=3, ls=" ", color="black", alpha=0.5)
        updater = polar_update if (polar) else update

    ani = FuncAnimation(fig, updater, frames=time_chunks, interval=10)
    plt.show()

