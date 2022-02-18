import numpy
import math
from matplotlib import pyplot

N = 50                                # Number of points in each direction
v = 10                                # Number of vortices
x_start, x_end = -2.0, 2.0            # x-direction boundaries
y_start, y_end = -2.0, 2.0            # y-direction boundaries
x = numpy.linspace(x_start, x_end, N)
y = numpy.linspace(y_start, y_end, N)
X, Y = numpy.meshgrid(x, y)
strength = 1.0
x_vortex = numpy.linspace(x_start, x_end, v)
y_vortex = numpy.zeros(v)

# compute the velocity field and psi on the mesh grid
u_vortex = numpy.zeros((N, N), dtype=float)
v_vortex = numpy.zeros((N, N), dtype=float)
for i in range(v):
    u_vortex = u_vortex + strength / (2 * math.pi) * (Y - y_vortex[i]) / ((X - x_vortex[i])**2 + (Y - y_vortex[i])**2)
    v_vortex = v_vortex + (-strength) / (2 * math.pi) * (X - x_vortex[i]) / ((X - x_vortex[i]) ** 2 + (Y - y_vortex[i])**2)

# plot the streamlines
pyplot.xlim(x_start, x_end)
pyplot.ylim(y_start, y_end)
pyplot.streamplot(X, Y, u_vortex, v_vortex, density=2, linewidth=1, arrowsize=1, arrowstyle='->')
pyplot.scatter(x_vortex, y_vortex, color='#CD2305', s=80, marker='o')
pyplot.show()

