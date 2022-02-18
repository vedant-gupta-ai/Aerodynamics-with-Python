import numpy as np
from matplotlib import pyplot
import math


def velocity(strength, X, Y, x, y):
    u = (strength / (2 * math.pi) * (X - x) / ((X - x) ** 2 + (Y - y) ** 2))
    v = (strength / (2 * math.pi) * (Y - y) / ((X - x) ** 2 + (Y - y) ** 2))
    return u, v


N = 10  #  Number of sources
p = 100  #  Number of grid points in each direction
strength = 0.5
u_inf = 1.0
x_start, x_end = -2.0, 2.0
y_start, y_end = -2.0, 2.0
x = np.linspace(x_start, x_end, p)
y = np.linspace(y_start, y_end, p)
X, Y = np.meshgrid(x, y)

#  Putting the values for the freestream
u_x = u_inf * np.ones((p, p), dtype=float)
u_y = u_inf * np.zeros((p, p), dtype=float)

#  Defining location of Sources along y-axis
y_src = np.linspace(-1.0, 1.0, N)
x_src = np.zeros(N, dtype=float)

#  Calculating velocity values for the mesh due to the sources
u_source = np.zeros((p, p), dtype=float)
v_source = np.zeros((p, p), dtype=float)

for i in range(N):
    u_source = u_source + (strength / (2 * np.pi) * (X - x_src[i]) / ((X - x_src[i]) ** 2 + (Y - y_src[i]) ** 2))
    v_source = v_source + (strength / (2 * np.pi) * (Y - y_src[i]) / ((X - x_src[i]) ** 2 + (Y - y_src[i]) ** 2))

# Total velocities
u_total = u_source + u_x
v_total = v_source + u_y
V = np.sqrt(u_total**2 + v_total**2)

#  Stagnation Point
sp1, sp2 = np.unravel_index(np.argmin(V, axis=None), V.shape)
print(sp1, sp2)

#  Plotting
pyplot.xlim(-2.0, 2.0)
pyplot.ylim(-2.0, 2.0)
pyplot.scatter(x_src, y_src, color='#CD2305', s=8, marker='o')
#  pyplot.scatter(sp1, sp2, color='#CD2305', s=8, marker='o')
pyplot.streamplot(X, Y, u_total, v_total, density=2, linewidth=1, color='black', arrowsize=1, arrowstyle='->')
pyplot.show()