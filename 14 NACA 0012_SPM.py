import numpy as np
from matplotlib import pyplot
from scipy import integrate
import math


########################################################################################################################
# CREATING PANEL CLASS

class Panel:
    def __init__(self, xa, ya, xb, yb):
        self.xa, self.ya = xa, ya
        self.xb, self.yb = xb, yb

        self.xc, self.yc = (xa+xb)/2, (ya+yb)/2  # Control Points

        self.length = ((xb-xa)**2+(yb-ya)**2)**0.5

        # Defining panel orientation
        if xb - xa <= 0.0:
            self.beta = math.acos((yb - ya) / self.length)
        elif xb - xa > 0.0:
            self.beta = math.pi + math.acos(-(yb - ya) / self.length)

        # Defining position on the panel as upper or lower
        if self.beta <= math.pi:
            self.loc = 'upper'
        else:
            self.loc = 'lower'

        # Initializing important parameters to be used later as zero for now
        self.sigma = 0.0
        self.vt = 0.0
        self.cp = 0.0


########################################################################################################################
# DEFINING FUNCTION TO CREATE PANEL

def panel_def(x, y, N):
    radius = (x.max() - x.min())/2
    centre = (x.max() + x.min())/2
    x_circle = centre + radius * np.cos(np.linspace(0.0, 2*np.pi, N+1))  # Defining the x coordinates of the circle

    x_ends = np.copy(x_circle)  # Copying the x coordinates of the circle to the airfoil
    y_ends = np.empty_like(x_ends)  # Initializing y cord as an empty array, to be filled by values from interpolation

    x, y = np.append(x, x[0]), np.append(y, y[0])  # Appending thr arrays so as to accommodate index [i+1]

    # Computing y coordinates
    r = 0
    for i in range(N):
        while r < len(x) - 1:
            if (x[r] <= x_ends[i] <= x[r+1]) or (x[r+1] <= x_ends[i] <= x[r]):
                break
            else:
                r += 1
        a = (y[r+1] - y[r]) / (x[r+1] - x[r])
        b = y[r+1] - a * x[r+1]
        y_ends[i] = a * x_ends[i] + b

    y_ends[N] = y_ends[0]

    # Defining panels
    p = np.empty(N, dtype=object)  # First defining an empty array
    for i in range(N):
        p[i] = Panel(x_ends[i], y_ends[i], x_ends[i+1], y_ends[i+1])

    return p


########################################################################################################################
# CALLING FILE

file_name = 'NACA 0012.dat'
x, y = np.loadtxt(file_name, dtype=float, delimiter='\t', unpack=True)

########################################################################################################################
# CREATING THE REQUIRED PANELS THROUGH FUNCTION

N = 40  # Number of panels in the geometry
panels = panel_def(x, y, N)


########################################################################################################################
# DEFINING COMMON FUNCTION FOR ALL INTEGRATIONS (sigma, Vt, u, v)

def integral(x, p, term1, y, term2):
    def integrand(s):
        return ((x - (p.xa - math.sin(p.beta)*s))*term1 +
                (y - (p.ya + math.cos(p.beta)*s))*term2)/((x - (p.xa - math.sin(p.beta) * s)) **2 +
                (y - (p.ya + math.cos(p.beta) * s)) **2)
    return integrate.quad(integrand, 0.0, p.length)[0]


########################################################################################################################
# STORING SIGMA VALUES

A = np.empty((N, N), dtype=float)
np.fill_diagonal(A, 0.5)

# Constructing matrix A
for i, p_i in enumerate(panels):
    for j, p_j in enumerate(panels):
        if i != j:
            A[i, j] = (0.5/math.pi) * integral(p_i.xc, p_j, math.cos(p_i.beta), p_i.yc, math.sin(p_i.beta))

u_inf = 1.0  # Freestream Velocity

# Computing the RHS values
b = - u_inf * np.cos([p.beta for p in panels])

# Solving the system
sigma = np.linalg.solve(A, b)

# Storing values for the objects using the class Panel
for i, panel in enumerate(panels):
    panel.sigma = sigma[i]

########################################################################################################################
# COMPUTING TANGENTIAL VELOCITIES

v_tang = np.empty((N, N), dtype=float)
np.fill_diagonal(v_tang, 0.0)

# Filling the values of the tangential velocities
for i, p_i in enumerate(panels):
    for j, p_j in enumerate(panels):
        if i != j:
            v_tang[i, j] = (0.5/math.pi) * integral(p_i.xc, p_j, -math.sin(p_i.beta), p_i.yc, math.cos(p_i.beta))

b2 = - u_inf * np.sin([panel.beta for panel in panels])
Vt = np.dot(v_tang, sigma) + b2

# Storing values into the objects of the class Panel
for i, p in enumerate(panels):
    p.vt = Vt[i]

########################################################################################################################
# COMPUTING VALUES OF Cp

for p in panels:
    p.cp = 1.0 - (p.vt / u_inf)**2

########################################################################################################################
# COMPUTING CARTESIAN VELOCITIES (u, v)

m = 20  # Number of the grid points

# Defining domain and grid
x_domain = np.linspace(-1.0, 2.0, m)
y_domain = np.linspace(-0.3, 0.3, m)
X, Y = np.meshgrid(x_domain, y_domain)
u_cartesian = np.zeros((m, m), dtype=float)
v_cartesian = np.zeros((m, m), dtype=float)

# Creating matrices of the freestream velocity components
u_x = u_inf * np.ones((m, m), dtype=float)
u_y = u_inf * np.zeros((m, m), dtype=float)

# Vectorizing the velocity components
vectorized_u = np.vectorize(integral)
vectorized_v = np.vectorize(integral)

# Storing the values
for i, p in enumerate(panels):
    u_cartesian = u_cartesian + (p.sigma / (2.0 * math.pi) * vectorized_u(X, p, 1.0, Y, 0.0))

for i, p in enumerate(panels):
    v_cartesian = v_cartesian + (p.sigma / (2.0 * math.pi) * p.sigma * vectorized_v(X, p, 0.0, Y, 1.0))

# Total velocities
u = u_cartesian + u_x
v = v_cartesian + u_y

########################################################################################################################
# PLOTTING

pyplot.xlim(-1.0, 2.0)
pyplot.ylim(-0.3, 0.3)
pyplot.plot(x, y)
# pyplot.plot([p.cp for p in panels])
pyplot.streamplot(X, Y, u, v, density=2, linewidth=1, color='blue', arrowstyle='->')
pyplot.fill_between(x, y, color='k')
pyplot.show()

########################################################################################################################