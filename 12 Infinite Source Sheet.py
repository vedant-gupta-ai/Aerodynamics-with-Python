import numpy as np
from matplotlib import pyplot
from scipy import integrate

p = 50  #  Number of grid points in each direction
strength = 20  #  Strength of the source sheet
u_inf = 10
x_start, x_end = -5.0, 5.0
y_start, y_end = -5.0, 5.0
x = np.linspace(x_start, x_end, p)
y = np.linspace(y_start, y_end, p)
X, Y = np.meshgrid(x, y)

#  Putting the values for the freestream
u_x = u_inf * np.ones((p, p), dtype=float)
u_y = u_inf * np.zeros((p, p), dtype=float)

#  Defining limits along y-axis for the sheet
y_min, y_max = -1.0, 1.0

#  Creating lambda functions for u and v
integrand_u = lambda s, x, y: x / (x**2 + (y - s)**2)
integrand_v = lambda s, x, y: (y - s) / (x**2 + (y - s)**2)


#  Dummy function that will be vectorized
def integration(x, y, integrand):
    return integrate.quad(integrand, y_min, y_max, args=(x, y))[0]


vectorized = np.vectorize(integration)

#  Computing the velocity components due to source sheet
u_ss = (strength / 2*np.pi) * vectorized(X, Y, integrand_u)
v_ss = (strength / 2*np.pi) * vectorized(X, Y, integrand_v)

#  Total values
u = u_ss + u_x
v = v_ss + u_y

#  Plotting
pyplot.xlim(x_start, x_end)
pyplot.ylim(y_start, y_end)
pyplot.axvline(0.0, (y_min - y_start) / (y_end - y_start), (y_max - y_start) / (y_end - y_start), color='#CD2305', linewidth=4)
pyplot.streamplot(X, Y, u, v, density=2, linewidth=1, color='blue', arrowsize=1, arrowstyle='->')
V = np.sqrt(u**2 + v**2)  # Total velocity
j_stagn, i_stagn = np.unravel_index(V.argmin(), V.shape)
pyplot.scatter(x[i_stagn], y[j_stagn], color='black', s=40, marker='D')    #  Plotting the stagnation point
pyplot.show()
