import numpy as np
from matplotlib import pyplot
import math

n = 51  # Mesh points
u_inf = 1
x_start, y_start = -1.0, -0.5
x_end, y_end = 2.0, 0.5
x = np.linspace(x_start, x_end, n)
y = np.linspace(y_start, y_end, n)
X, Y = np.meshgrid(x, y)  # Generating the base mesh

# Calling all the values in the resources
file_x = 'NACA 0012_x.txt'
file_y = 'NACA 0012_y.txt'
file_strength = 'NACA 0012_strength.txt'
af_x = np.loadtxt(file_x)
af_y = np.loadtxt(file_y)
af_strn = np.loadtxt(file_strength)

# Freestream values
u_x = u_inf * np.ones((n, n), dtype=float)
u_y = u_inf * np.zeros((n, n), dtype=float)

# Sources' values
u_source = np.zeros((n, n), dtype=float)
for i in range(len(af_x)):
    u_source = u_source + (af_strn[i] / (2 * math.pi) * (X - af_x[i]) / ((X - af_x[i])**2 + (Y - af_y[i])**2))

v_source = np.zeros((n, n), dtype=float)
for i in range(len(af_y)):
    v_source = v_source + (af_strn[i] / (2 * math.pi) * (Y - af_y[i]) / ((X - af_x[i])**2 + (Y - af_y[i])**2))

# Total values
u = u_x + u_source
v = u_y + v_source

cp = 1 - (u**2 + v**2)/u_inf**2  # Coefficient of pressure
print(np.max(cp))
print(np.unravel_index(np.argmax(cp, axis=None), cp.shape))

# Plotting
pyplot.xlim(-1.0, 2.0)
pyplot.ylim(-0.5, 0.5)
pyplot.scatter(af_x, af_y, color='#CD2305', s=8, marker='o')
pyplot.streamplot(X, Y, u, v, density=2, arrowsize=1, arrowstyle='->')
pyplot.show()

