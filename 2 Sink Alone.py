import numpy as np
from matplotlib import pyplot
import math

sink_str = -5
x_sink, y_sink = 0,0
x_arr = np.arange(-5,5,0.1) # Range from -5 to 5 with step size of 0.1
y_arr = np.arange(-5,5,0.1)
X, Y = np.meshgrid(x_arr, y_arr) # Creating the uniform grid

# Setting up the velocities
u = (sink_str*(x_arr - x_sink))/((2 * math.pi)*((x_arr - x_sink)**2 + (y_arr - y_sink)**2))
v = (sink_str*(y_arr - y_sink))/((2 * math.pi)*((x_arr - x_sink)**2 + (y_arr - y_sink)**2))

# Forming the streamplot
pyplot.scatter(x_sink, y_sink)
pyplot.xlim(-5, 5)
pyplot.ylim(-5, 5)
pyplot.xlabel("X")
pyplot.ylabel("Y")
pyplot.streamplot(X, Y, u, v, density=2, arrowsize=1, arrowstyle='->')
pyplot.show()
