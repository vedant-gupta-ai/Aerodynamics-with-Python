import numpy as np
from matplotlib import pyplot
import math

vel_inf = 2.0
n = 100 # Number of steps
sink_str = -5.0
x_sink, y_sink = 0,0
x_arr = np.linspace(-5.0,5.0,n) # Range from -5 to 5 with step size of 0.1
y_arr = np.linspace(-5.0,5.0,n)
X, Y = np.meshgrid(x_arr, y_arr) # Creating the uniform grid

# Free stream velocity components and psi
psi_inf = vel_inf * Y # Stream function of uniform flow = Uy for horizontal flow
u_inf = vel_inf * np.ones((n,n), dtype=float)
v_inf = vel_inf * np.zeros((n,n), dtype=float)

# Setting up the velocities and psi
psi_s = (sink_str/2*math.pi)*np.arctan((Y - y_sink)/(X - x_sink)) # Stream function for sink
u = (sink_str*(X - x_sink))/((2 * math.pi)*((X - x_sink)**2 + (Y - y_sink)**2)) + u_inf
v = (sink_str*(Y - y_sink))/((2 * math.pi)*((X - x_sink)**2 + (Y - y_sink)**2)) + v_inf

# Forming the streamplot
psi = psi_inf + psi_s # Overall stream function
pyplot.scatter(x_sink, y_sink)
pyplot.xlim(-5.0, 5.0)
pyplot.ylim(-5.0, 5.0)
pyplot.xlabel("X")
pyplot.ylabel("Y")
pyplot.streamplot(X, Y, u, v, density=2, arrowsize=1, arrowstyle='->')
pyplot.contour(X, Y, psi,levels=[sink_str, -sink_str], colors='#CD2305', linewidths=2, linestyles='solid')
pyplot.show()
