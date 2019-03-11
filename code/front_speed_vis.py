import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


with np.load('front_speed_data.npz') as data:
    E = data['E']
    rho = 900
    v = data['speeds']
    fs = data['front_speeds']

    for k in range(len(E)):
        plt.plot(v, fs[k, :], 'bs')
        p = np.polyfit(v, fs[k, :], deg=1)
        print(p)
        print(p[1]/np.sqrt(E[k]/rho))
        print('\n')
    plt.grid()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X, Y, Z = np.sqrt(E/rho), v, fs
    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
    ax.set_xlabel(r'$\sqrt{\frac{E}{\rho}}$')
    ax.set_ylabel(r'$v_{strike}$')
    plt.show()

