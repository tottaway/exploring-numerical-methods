import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


# constants
c = 0.05
T = 100
X, Y = 1.5, 1.5

Nt = 4500
Nx = 75
Ny = 75

dt = T / Nt
dx = X / Nx
dy = Y / Ny

X_values = np.arange(-X/2, X/2, dx)
Y_values = np.arange(-Y/2, Y/2, dy)
X_values, Y_values = np.meshgrid(X_values, Y_values)


def wave(Nt, Nx, Ny, U0):
    # initialize variables
    U = np.zeros((Nt, Nx, Ny))
    Uxx = np.zeros((Nt, Nx, Ny))
    Uyy = np.zeros((Nt, Nx, Ny))
    Ut = np.zeros((Nt, Nx, Ny))
    U[0, :, :] = U0
    for t in range(1, Nt):
        U[t-1, 0, :] *= 0
        U[t-1, Nx-1, :] *= 0
        U[t-1, :, 0] *= 0
        U[t-1, :, Ny-1] *= 0

        def calc_Uxx(U_1):
            res = np.zeros((Nx, Ny))
        
            curr_slice = U_1[1:Nx-1, :]
            forward = U_1[2:Nx, :]
            backward = U_1[0:Nx-2, :]

            res[1:Nx-1, :] = (forward - 2*curr_slice + backward) / (dx*dx)

            return res
        Uxx[t, :, :] = calc_Uxx(U[t-1, :, :])
        # calculate Uyy 
        def calc_Uyy(U_1):
            res = np.zeros((Nx, Ny))
        
            curr_slice = U_1[1:Ny-1, :]
            forward = U_1[2:Ny, :]
            backward = U_1[0:Ny-2, :]

            res[1:Ny-1, :] = (forward - 2*curr_slice + backward) / (dy*dy)

            return res
        Uyy[t, :, :] = calc_Uyy(U[t-1, :, :])

        for y in range(1, Ny-1):
            curr_slice = U[t-1, :, y]
            forward = U[t-1, :, y+1]
            backward = U[t-1, :, y-1]

            Uyy[t, :, y] = (forward - 2*curr_slice + backward) / (dy*dy)
        
        def Utt(Uxx, Uyy):
            return c**2 * (Uxx + Uyy)

        def rk4(Uxx, Uyy, Ut0, f):
            k1 = f(Ut0)
            k2 = f(Ut0+k1*0.5*dt)
            k3 = f(Ut0+0.5*k2*dt)
            k4 = f(Ut0+k3*dt)
            
            delta = dt * (k1 + 2*k2 + 2*k3 + k4) /6
            return Ut0 + delta

        Ut_1 = Ut[t-1, :, :]

        def calc_Ut(Ut0):
            k1 = Utt(Uxx[t-1, :, :], Uyy[t-1, :, :])
            k2 = Utt(calc_Uxx(Ut0+0.5*k1*dt),
                   calc_Uyy(Ut0+0.5*k1*dt))
            k3 = Utt(calc_Uxx(Ut0+0.5*k2*dt),
                   calc_Uyy(Ut0+0.5*k2*dt))
            k4 = Utt(calc_Uxx(Ut0+k3*dt),
                   calc_Uyy(Ut0+k3*dt))
            return (k1 + 2*k2 + 2*k3 + k4) /6

        Ut[t, :, :] = rk4(Uxx[t, :, :], Uyy[t, :, :], Ut_1, calc_Ut)


        U[t, :, :] = U[t-1, :, :] + dt * Ut[t, :, :]
    
        print(t, U[t, int(Nx/2), int(Ny/2)])

    # initialize plot
    fig = plt.figure()

    plt.title("Animation of wave equation")
    ax = fig.add_subplot(111, projection='3d')
    zs = U[0, :, :]
    ax.set_zlim(-1, 1)
    surface = None
    tstart = time.time()
    for t in range(0, Nt, 80):
        # If a line collection is already remove it before drawing.
        if surface:
            ax.collections.remove(surface)

        # Plot the new wireframe and pause briefly before continuing.
        Z = U[t, :, :]
        surface = ax.plot_surface(X_values, Y_values, Z, cmap=cm.coolwarm, linewidth=10,
          antialiased=False)
        print(t, np.sum(np.sum(Z-U[t-1, :, :])))
        plt.pause(.0001)

    # anim = FuncAnimation(fig, update, frames=1000, interval=1)

U0 = np.sin(20*(1 / (np.sqrt(X_values**2 + Y_values**2) + 1) - 0.5))
wave(Nt, Nx, Ny, U0)