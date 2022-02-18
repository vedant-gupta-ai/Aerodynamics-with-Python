import numpy as np
from matplotlib import pyplot
import math

x_dblt, y_dblt = 0,0
u_inf = 1
k = 1 # Strength of the doublet
n = 100
x = np.linspace(-2.0, 2.0, n)
y = np.linspace(-2.0, 2.0, n)
X, Y = np.meshgrid(x, y)


def freestream(u_inf, n, Y):
    u_x = u_inf*np.ones((n,n), dtype=float)
    u_y = u_inf*np.zeros((n,n), dtype=float)
    psi_freestream = u_inf*Y
    return u_x, u_y, psi_freestream


def dblt_vel(k, X, Y):
    u = (- k / (2 * math.pi))*(X**2-Y**2) /((X**2+Y**2)**2)
    v = (- k / (2 * math.pi))*2*X*Y /((X**2+Y**2)**2)
    psi_dblt = (-k/(2*math.pi))*(Y)/(X**2+Y**2)
    return u, v, psi_dblt


u_free, v_free, psi_free = freestream(u_inf, n, Y)
u_db, v_db, psi_db = dblt_vel(k, X, Y,)

# Overall stuff
psi = psi_db + psi_free
u = u_free + u_db
v = v_free + v_db

cp = 1 - (u**2 + v**2)/u_inf**2

x_stagn1, y_stagn1 = +math.sqrt(k / (2 * math.pi * u_inf)), 0.0
x_stagn2, y_stagn2 = -math.sqrt(k / (2 * math.pi * u_inf)), 0.0

# Plotting
pyplot.xlim(-2.0, 2.0)
pyplot.ylim(-2.0, 2.0)
pyplot.scatter(x_dblt, y_dblt, color='#CD2305', s=8, marker='o')
cntf = pyplot.contourf(X, Y, cp, levels=np.linspace(-2.0, 2.0, 100), extend='both')
cbar=pyplot.colorbar(cntf)
cbar.set_ticks([-2.0, -1.0, 0.0, 1.0])
pyplot.contour(X, Y, psi, levels=[0.], colors='#CD2305', linewidths=2, linestyles='solid')
pyplot.scatter([x_stagn1, x_stagn2], [y_stagn1, y_stagn2], color='g', s=8, marker='o')
pyplot.show()
