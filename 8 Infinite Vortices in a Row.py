import numpy
from matplotlib import pyplot
import math

N = 50                                # Number of points in each direction
a = 0.1
x_start, x_end = -2.0, 2.0            # x-direction boundaries
y_start, y_end = -2.0, 2.0            # y-direction boundaries
x = numpy.linspace(x_start, x_end, N)
y = numpy.linspace(y_start, y_end, N)
X, Y = numpy.meshgrid(x, y)
strength = 5.0

u = (strength / 2 * a) * (numpy.sinh(2 * math.pi * Y / a))/(numpy.cosh(2 * math.pi * Y / a)-numpy.cos(2 * math.pi * X / a))
v = (-strength / 2 * a) * (numpy.sinh(2 * math.pi * X / a))/(numpy.cosh(2 * math.pi * Y / a)-numpy.cos(2 * math.pi * X / a))

pyplot.xlim(x_start, x_end)
pyplot.ylim(y_start, y_end)
pyplot.streamplot(X, Y, u, v, density=2, linewidth=1, arrowsize=1, arrowstyle='->')
pyplot.show()