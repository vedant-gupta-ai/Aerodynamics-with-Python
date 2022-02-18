import numpy as np
from matplotlib import pyplot
from scipy import integrate
import math


########################################################################################################################
# STARTING OUT BY DEFINING A CLASS THAT WILL ALLOW US TO DEFINE ALL THE PANELS IN THE GEOMETRY

class Panels:
    def __init__(self, xa, xb, ya, yb):
        self.xa, self.xb = xa, xb
        self.ya, self.yb = ya, yb
        self.xc, self.yc = (xa + xb) / 2, (ya + yb) / 2  # Control-Point (center-point)
        self.length = math.sqrt((xb - xa) ** 2 + (yb - ya) ** 2)  # Length of the panel

        #  Panel orientation
        if xb - xa <= 0.0:
            self.beta = math.acos((yb - ya) / self.length)
        elif xb - xa > 0.0:
            self.beta = math.pi + math.acos(-(yb - ya) / self.length)

        # Initializing the values for now, would be changed later in the code
        self.strength = 0.0
        self.vt = 0.0
        self.cp = 0.0


########################################################################################################################
# DEFINING THE CYLINDER

u_inf = 1.0
R = 1.0  # Radius of the cylinder
x_centre, y_centre = 0.0, 0.0
theta = np.linspace(0.0, 2.0 * np.pi, 100)
x_cyl, y_cyl = x_centre + R * np.cos(theta), y_centre + R * np.sin(theta)

########################################################################################################################
#  DEFINING THE PANELS AND ITS GEOMETRY USING THE CLASS CREATED ABOVE

N = 10  # Number of panels
panels = np.empty(N, dtype=object)  # dtype is object as the method panel of the class Panel is an object
theta_panel = np.linspace(0.0, 2*np.pi, N + 1)
x_ends = R * np.cos(theta_panel)
y_ends = R * np.sin(theta_panel)
for i in range(N):
    panels[i] = Panels(x_ends[i], x_ends[i+1], y_ends[i], y_ends[i+1])  # Panels defined using class Panels

########################################################################################################################
# COMPUTING SOURCE MATRIX [A] THROUGH THE BOUNDARY CONDITION OF ZERO NORMAL VELOCITY
# ZERO NORMAL VELOCITY GIVES TWO CONDITIONS, 0.5 FOR i=j, AND AN INTEGRAL TERM FOR OTHER CASES


def integral_normal(p_i, p_j):
    def integrand(s):
        return (((p_i.xc - (p_j.xa - math.sin(p_j.beta) * s)) * math.cos(p_i.beta) +
                 (p_i.yc - (p_j.ya + math.cos(p_j.beta) * s)) * math.sin(p_i.beta)) /
                ((p_i.xc - (p_j.xa - math.sin(p_j.beta) * s)) ** 2 +
                 (p_i.yc - (p_j.ya + math.cos(p_j.beta) * s)) ** 2))
    return integrate.quad(integrand, 0.0, p_j.length)[0]


# Computing the source influence matrix
A = np.empty((N, N), dtype=float)
np.fill_diagonal(A, 0.5)

for i, p_i in enumerate(panels):
    for j, p_j in enumerate(panels):
        if i != j:
            A[i, j] = 0.5 / math.pi * integral_normal(p_i, p_j)

# Computing the RHS
b = - u_inf * np.cos([p.beta for p in panels])

# Solving the linear system
sigma = np.linalg.solve(A, b)

for i, panel in enumerate(panels):
    panel.sigma = sigma[i]  # Source strengths obtained


########################################################################################################################
# COMPUTING THE TANGENTIAL VELOCITIES AT EACH CONTROL POINT

def integral_tangential(p_it, p_jt):
    def integrand(s):
        return ((-(p_it.xc - (p_jt.xa - math.sin(p_jt.beta) * s)) * math.sin(p_it.beta) +
                 (p_it.yc - (p_jt.ya + math.cos(p_jt.beta) * s)) * math.cos(p_it.beta)) /
                ((p_it.xc - (p_jt.xa - math.sin(p_jt.beta) * s)) ** 2 +
                 (p_it.yc - (p_jt.ya + math.cos(p_jt.beta) * s)) ** 2))
    return integrate.quad(integrand, 0.0, p_jt.length)[0]


# Computing the matrix of the linear system for tangential velocities
V = np.empty((N, N), dtype=float)
np.fill_diagonal(V, 0.0)

for i, p_i in enumerate(panels):
    for j, p_j in enumerate(panels):
        if i != j:
            V[i, j] = 0.5 / math.pi * integral_tangential(p_i, p_j)

# Computing the RHS
b = - u_inf * np.sin([panel.beta for panel in panels])

# Computing the tangential velocity at each panel center-point
vt = np.dot(V, sigma) + b

for i, panel in enumerate(panels):
    panel.vt = vt[i]

########################################################################################################################
# CALCULATING Cp

for panel in panels:
    panel.cp = 1.0 - (panel.vt / u_inf)**2

########################################################################################################################
# COMPUTING u & v VELOCITIES IN THE CARTESIAN DOMAIN


def integrate_cartesian_u(x, y, p_jc):
    def u_ss_cart(s):
        return (x - (p_jc.xa - math.sin(p_jc.beta) * s)) / ((x - (p_jc.xa - math.sin(p_jc.beta) * s)) ** 2 +
                                                            (y - (p_jc.ya + math.cos(p_jc.beta) * s)) ** 2)

    return integrate.quad(u_ss_cart, 0.0, p_jc.length)[0]


def integrate_cartesian_v(x, y, p_jc):
    def v_ss_cart(s):
        return (y - (p_jc.ya + math.cos(p_jc.beta) * s)) / ((x - (p_jc.xa - math.sin(p_jc.beta) * s)) ** 2 +
                                                            (y - (p_jc.ya + math.cos(p_jc.beta) * s)) ** 2)

    return integrate.quad(v_ss_cart, 0.0, p_jc.length)[0]


m = 50  # Number of mesh points
x_domain = np.linspace(-3.0, 3.0, m)
y_domain = np.linspace(-3.0, 3.0, m)
X, Y = np.meshgrid(x_domain, y_domain)
u_x = u_inf * np.ones((m, m), dtype=float)
u_y = u_inf * np.zeros((m, m), dtype=float)

# Calculating the velocities on the grid points
u_cartesian = np.zeros((m, m), dtype=float)
v_cartesian = np.zeros((m, m), dtype=float)

# Vectorizing the velocity arrays to achieve quantities comparable with mesh
vectorized_u = np.vectorize(integrate_cartesian_u)
vectorized_v = np.vectorize(integrate_cartesian_v)

for i, p_j in enumerate(panels):
    u_cartesian = u_cartesian + (0.5/np.pi) * p_j.sigma * vectorized_u(X, Y, p_j)

for i, p_j in enumerate(panels):
    v_cartesian = v_cartesian + (0.5/np.pi) * p_j.sigma * vectorized_v(X, Y, p_j)

# Computing total velocities
u = u_cartesian + u_x
v = v_cartesian + u_y

########################################################################################################################
# PLOTTING

pyplot.xlim(-3.0, 3.0)
pyplot.ylim(-3.0, 3.0)
# pyplot.quiver(X, Y, u, v)
pyplot.streamplot(X, Y, u, v, density=2, linewidth=1, color='blue', arrowsize=1, arrowstyle='->')
pyplot.fill_between(x_cyl, y_cyl, color='blue')  # Fills colour in between two curves
pyplot.plot(x_cyl, y_cyl)

# centre = 0.0, 0.0
# c = pyplot.Circle(centre, radius=1)
# pyplot.gca().add_patch(c)  # Fills the spot, will work only for defined libraries

# pyplot.scatter([p.xc for p in panels], [p.cp for p in panels]) #  Sets a point on the location of every centre point
# pyplot.plot(x_ends, y_ends)  # Plots locations of every panel end points
# pyplot.scatter([panels.xc for panels in panels], [panels.yc for panels in panels], marker='*', c='black')
# Plots panel end points

pyplot.show()

########################################################################################################################