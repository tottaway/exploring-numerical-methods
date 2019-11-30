import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
"""
About this file:
Runs a simulation of the wave equation in two dimensions

Things to explore:
* Change initial condition matrix
* Change size of observed area using X, Y (Note, will increase runtime drastically)
* Forced oscillations using the force matrix in Utt
* Changing number in time_chunks 
* Change c which corresponds to wave speed (raises instabillity)
* Raise and lower Nt to examine stabilty of solutions
"""
c = 0.005
T = 45
X, Y = 4, 4

Nt = 3500
time_chunks = int(T*2)                 # number of frames computed
time_chunk_size = int(Nt/time_chunks)   # time steps to compute each frame
Nx = X * 35 
Ny = Y * 35

dt = T/time_chunk_size
dx = X / Nx
dy = Y / Ny

X_values = np.arange(-X/2, X/2, dx)
Y_values = np.arange(-Y/2, Y/2, dy)
X_values, Y_values = np.meshgrid(X_values, Y_values)

initial_cond = np.zeros((Nx, Ny)) #np.sin(10 / ((np.sqrt(X_values**2 + Y_values**2) + 1) - 0.5))

hold_boundary_at_zero = True

# wave equation
def Utt(Uxx, Uyy, t):
    """Wave equation: set time dependent forces using 'force' matrix"""
    force = np.zeros((Nx, Ny))
    force[int(3*Nx/4), int(3*Ny/4)] = 0.1*np.cos(t)
    return c**2 * (Uxx + Uyy) + force

def calc_Uxx(U_1):
    res = np.zeros((Nx, Ny))

    curr_slice = U_1[1:Nx-1, :]
    forward = U_1[2:Nx, :]
    backward = U_1[0:Nx-2, :]

    res[1:Nx-1, :] = (forward - 2*curr_slice + backward) / (dx**2)

    return res

def calc_Uyy(U_1):
    res = np.zeros((Nx, Ny))

    curr_slice = U_1[:, 1:Ny-1]
    forward = U_1[:, 2:Ny]
    backward = U_1[:, 0:Ny-2]

    res[:, 1:Ny-1] = (forward - 2*curr_slice + backward) / (dy**2)

    return res


def wave(Nt, Nx, Ny, U0, boundary):
    # initialize variables
    U = np.zeros((time_chunks, Nx, Ny))
    U[0, :, :] = U0
    
    Ut = np.zeros((time_chunks, Nx, Ny))

    # step through time and compute frames to be displayed
    for time_chunk in tqdm(range(1, time_chunks)):
        U_curr = np.zeros((time_chunk_size, Nx, Ny))
        Uxx = np.zeros((time_chunk_size, Nx, Ny))
        Uyy = np.zeros((time_chunk_size, Nx, Ny))
        Ut_curr = np.zeros((time_chunk_size, Nx, Ny))
        
        U_curr[0, :, :] = U[time_chunk-1, :, :]
        Ut_curr[0, :, :] = Ut[time_chunk-1, :, :]
        
        for t in range(1, time_chunk_size):
            Ut_1 = Ut_curr[t-1, :, :]

            midpoint = U_curr[t-1, :, :] + Ut_1*0.5*dt
            curr_time = (time_chunk*time_chunk_size + t)/(Nt/T)  
            Utt_k2 = Utt(calc_Uxx(midpoint), calc_Uyy(midpoint), curr_time)

            Ut_curr[t, :, :] = dt * (Utt_k2) + Ut_1
            U_curr[t, :, :] = U_curr[t-1, :, :] + dt * Ut_curr[t, :, :]

            if boundary:
                U_curr[t, 0, :] *= 0
                U_curr[t, Nx-1, :] *= 0
                U_curr[t, :, 0] *= 0
                U_curr[t, :, Ny-1] *= 0

        U[time_chunk, :, :] = U_curr[-1, :, :]
        Ut[time_chunk, :, :] = Ut_curr[-1, :, :]


    # create animation
    fig = plt.figure()

    plt.title("Animation of wave equation")
    ax = fig.add_subplot(111, projection='3d')
    zs = U[0, :, :]
    ax.set_zlim(-4, 4)
    surface = None
    tstart = time.time()
    for t in range(0, time_chunks):
        if surface:
            ax.collections.remove(surface)

        Z = U[t, :, :]
        surface = ax.plot_surface(X_values, Y_values, Z, cmap="Blues", linewidth=10,
          antialiased=False)
        plt.pause(.0001)


if __name__=="__main__":
    wave(Nt, Nx, Ny, initial_cond, hold_boundary_at_zero)
