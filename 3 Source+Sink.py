import numpy as np
from matplotlib import pyplot
import math

N = 100                                # number of points in each direction
x_start, x_end = -5.0, 5.0            # boundaries in the x-direction
y_start, y_end = -5.0, 5.0            # boundaries in the y-direction
x = np.linspace(x_start, x_end, N)    # creates a 1D-array with the x-coordinates
y = np.linspace(y_start, y_end, N)    # creates a 1D-array with the y-coordinates

X, Y = np.meshgrid(x, y)              # generates a mesh grid

strength_source = 5.0                      # source strength
x_source, y_source = 0.0, 2.0             # location of the source

# compute the velocity field on the mesh grid
u_source = (strength_source / (2 * math.pi) *
            (X - x_source) / ((X - x_source)**2 + (Y - y_source)**2))
v_source = (strength_source / (2 * math.pi) *
            (Y - y_source) / ((X - x_source)**2 + (Y - y_source)**2))

sink_str = -5
x_sink, y_sink = 0,-2.0
x_arr = np.arange(-5,5,0.1) # Range from -5 to 5 with step size of 0.1
y_arr = np.arange(-5,5,0.1)

# Setting up the velocities
u_sink = (sink_str*(X - x_sink))/((2 * math.pi)*((X - x_sink)**2 + (Y - y_sink)**2))
v_sink = (sink_str*(Y - y_sink))/((2 * math.pi)*((X - x_sink)**2 + (Y - y_sink)**2))

u_pair = u_source + u_sink
v_pair = v_source + v_sink

pyplot.xlabel('x', fontsize=16)
pyplot.ylabel('y', fontsize=16)
pyplot.xlim(x_start, x_end)
pyplot.ylim(y_start, y_end)
pyplot.streamplot(X, Y, u_pair, v_pair,
                  density=2.0, linewidth=1, arrowsize=2, arrowstyle='->')
pyplot.scatter([x_source, x_sink], [y_source, y_sink],
               color='#CD2305', s=80, marker='o');
pyplot.show()
