import numpy as np
from matplotlib import pyplot
import math

c = 1
Nr = 100  # Step points in radial direction
radius = 1.15
r_begin, r_end = radius, 5.0
r = np.linspace(r_begin, r_end, Nr)

Nt = 145  # Step points for theta direction
t_begin, t_end = 0.0, 2.0 * np.pi
x_c = -0.15
y_c = 0.0
theta = np.linspace(t_begin, t_end, Nt)

R, T = np.meshgrid(r, theta)

z = R * np.exp(T * 1j)

zeta = z + c**2/z


#Plotting
pyplot.xlim(-2.0, 2.0)
pyplot.ylim(-2.0, 2.0)
ax = pyplot.subplot(111, polar=True)
ax.scatter(T, R, marker='|', c='black', linewidth=0.4)
pyplot.show()
