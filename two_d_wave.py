import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


# constants
c = 0.05
T = 0.75
X, Y = 2, 2

Nt = 6000
time_chunks = int(T*180)
time_chunk_size = int(Nt/time_chunks)
Nx = 100
Ny = 100

dt = T/time_chunk_size
dx = X / Nx
dy = Y / Ny

X_values = np.arange(-X/2, X/2, dx)
Y_values = np.arange(-Y/2, Y/2, dy)
X_values, Y_values = np.meshgrid(X_values, Y_values)


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

# wave equation
def Utt(Uxx, Uyy):
    return c**2 * (Uxx + Uyy)

def wave(Nt, Nx, Ny, U0):
    # initialize variables
    U = np.zeros((time_chunks, Nx, Ny))
    U[0, :, :] = U0
    
    Ut = np.zeros((time_chunks, Nx, Ny))

    for time_chunk in range(1, time_chunks):
        U_curr = np.zeros((time_chunk_size, Nx, Ny))
        Uxx = np.zeros((time_chunk_size, Nx, Ny))
        Uyy = np.zeros((time_chunk_size, Nx, Ny))
        Ut_curr = np.zeros((time_chunk_size, Nx, Ny))
        
        U_curr[0, :, :] = U[time_chunk-1, :, :]
        Ut_curr[0, :, :] = Ut[time_chunk-1, :, :]
        
        for t in range(1, time_chunk_size):
            # Uxx[t, :, :] = calc_Uxx(U_curr[t-1, :, :])
            # calculate Uyy 
            # Uyy[t, :, :] = calc_Uyy(U_curr[t-1, :, :])
            
            # def rk4(Uxx, Uyy, Ut0, f):
            #     k1 = f(Ut0)
            #     k2 = f(Ut0+k1*0.5*dt)
            #     k3 = f(Ut0+0.5*k2*dt)
            #     k4 = f(Ut0+k3*dt)
                
            #     delta = dt * (k1 + 2*k2 + 2*k3 + k4) /6
            #     return Ut0 + delta


            # def calc_Ut(Ut0):
            #     k1 = Utt(Uxx[t-1, :, :], Uyy[t-1, :, :])
            #     k2 = Utt(calc_Uxx(U_curr[t-1, :, :]+0.5*k1*dt),
            #         calc_Uyy(U_curr[t-1, :, :]+0.5*k1*dt))
            #     k3 = Utt(calc_Uxx(U_curr[t-1, :, :]+0.5*k2*dt),
            #         calc_Uyy(U_curr[t-1, :, :]+0.5*k2*dt))
            #     k4 = Utt(calc_Uxx(U_curr[t-1, :, :]+k3*dt),
            #         calc_Uyy(U_curr[t-1, :, :]+k3*dt))
            #     return (k1 + 2*k2 + 2*k3 + k4) /6

            # Ut[t, :, :] = rk4(Uxx[t, :, :], Uyy[t, :, :], Ut_1, calc_Ut)
            
            Ut_1 = Ut_curr[t-1, :, :]
            midpoint = U_curr[t-1, :, :] + Ut_1*0.5*dt
            Utt_k2 = Utt(calc_Uxx(midpoint), calc_Uyy(midpoint))
            # Ut_k1 = dt * (Utt_k1) + Ut_1
            # Utt_k2 = Utt(calc_Uxx(midpoint), calc_Uyy(midpoint))
            Ut_curr[t, :, :] = dt * (Utt_k2) + Ut_1
            U_curr[t, :, :] = U_curr[t-1, :, :] + dt * Ut_curr[t, :, :] #dt * Ut[t, :, :]
            # print(U_curr[:, int(Nx/2), int(Ny/2)])


            U_curr[t, 0, :] *= 0
            U_curr[t, Nx-1, :] *= 0
            U_curr[t, :, 0] *= 0
            U_curr[t, :, Ny-1] *= 0

        U[time_chunk, :, :] = U_curr[-1, :, :]
        Ut[time_chunk, :, :] = Ut_curr[-1, :, :]
        print(time_chunk, U[time_chunk, int(Nx/2), int(Ny/2)])
    
    # initialize plot
    fig = plt.figure()

    plt.title("Animation of wave equation")
    ax = fig.add_subplot(111, projection='3d')
    zs = U[0, :, :]
    ax.set_zlim(-1.5, 1.5)
    surface = None
    tstart = time.time()
    for t in range(0, Nt):
        # If a line collection is already remove it before drawing.
        if surface:
            ax.collections.remove(surface)

        # Plot the new wireframe and pause briefly before continuing.
        Z = U[t, :, :]
        surface = ax.plot_surface(X_values, Y_values, Z, cmap="Blues", linewidth=10,
          antialiased=False)
        print(t, np.sum(np.sum(Z-U[t-1, :, :])))
        plt.pause(.0001)

    # anim = FuncAnimation(fig, update, frames=1000, interval=1)

U0 = np.sin(10*(1 / (np.sqrt(X_values**2 + Y_values**2) + 1) - 0.5))
wave(Nt, Nx, Ny, U0)