import numpy
from matplotlib import pyplot
from scipy import integrate
import math


########################################################################################################################
# CREATING PANEL CLASS

class Panel:
    def __init__(self, xa, ya, xb, yb):
        self.xa, self.ya = xa, ya
        self.xb, self.yb = xb, yb

        self.xc, self.yc = (xa + xb) / 2, (ya + yb) / 2
        self.length = numpy.sqrt((xb - xa) ** 2 + (yb - ya) ** 2)

        if xb - xa <= 0.0:
            self.beta = numpy.arccos((yb - ya) / self.length)
        elif xb - xa > 0.0:
            self.beta = numpy.pi + numpy.arccos(-(yb - ya) / self.length)

        if self.beta <= numpy.pi:
            self.loc = 'upper'
        else:
            self.loc = 'lower'

        self.sigma = 0.0
        self.vt = 0.0
        self.cp = 0.0


########################################################################################################################
# DEFINING FUNCTION TO CREATE PANEL

def define_panels(x, y, N=40):
    R = (x.max() - x.min()) / 2.0  # circle radius
    x_center = (x.max() + x.min()) / 2.0  # x-coordinate of circle center

    theta = numpy.linspace(0.0, 2.0 * numpy.pi, N + 1)  # array of angles
    x_circle = x_center + R * numpy.cos(theta)  # x-coordinates of circle

    x_ends = numpy.copy(x_circle)  # x-coordinate of panels end-points
    y_ends = numpy.empty_like(x_ends)  # y-coordinate of panels end-points

    # extend coordinates to consider closed surface
    x, y = numpy.append(x, x[0]), numpy.append(y, y[0])

    # compute y-coordinate of end-points by projection
    I = 0
    for i in range(N):
        while I < len(x) - 1:
            if (x[I] <= x_ends[i] <= x[I + 1]) or (x[I + 1] <= x_ends[i] <= x[I]):
                break
            else:
                I += 1
        a = (y[I + 1] - y[I]) / (x[I + 1] - x[I])
        b = y[I + 1] - a * x[I + 1]
        y_ends[i] = a * x_ends[i] + b
    y_ends[N] = y_ends[0]

    # create panels
    panels = numpy.empty(N, dtype=object)
    for i in range(N):
        panels[i] = Panel(x_ends[i], y_ends[i], x_ends[i + 1], y_ends[i + 1])

    return panels


########################################################################################################################
# CALLING FILE

file_name = 'NACA 0012.dat'
x, y = numpy.loadtxt(file_name, dtype=float, delimiter='\t', unpack=True)


########################################################################################################################
# CREATING THE REQUIRED PANELS THROUGH FUNCTION

N = 40  # Number of panels in the geometry
panels = define_panels(x, y, N)


########################################################################################################################
# FREESTREAM CONDITIONS

class Freestream:
    def __init__(self, alpha, u_inf=1.0):
        self.u_inf = u_inf
        self.alpha = numpy.radians(alpha)  # degrees to radians


# define freestream conditions
freestream = Freestream(alpha=6.0, u_inf=1.0)


########################################################################################################################
# DEFINING COMMON FUNCTION FOR ALL INTEGRATIONS (sigma, Vt, u, v)

def integral(x, y, panel, dxdk, dydk):

    def integrand(s):
        return (((x - (panel.xa - numpy.sin(panel.beta) * s)) * dxdk +
                 (y - (panel.ya + numpy.cos(panel.beta) * s)) * dydk) /
                ((x - (panel.xa - numpy.sin(panel.beta) * s)) ** 2 +
                (y - (panel.ya + numpy.cos(panel.beta) * s)) ** 2))
    return integrate.quad(integrand, 0.0, panel.length)[0]


########################################################################################################################
# DEFINING THE SOURCE STRENGTH MATRIX, DERIVED FROM THE FLOW TANGENCY CONDITION

def source_contribution_normal(panels):
    A = numpy.empty((panels.size, panels.size), dtype=float)
    # source contribution on a panel from itself
    numpy.fill_diagonal(A, 0.5)
    # source contribution on a panel from others
    for i, panel_i in enumerate(panels):
        for j, panel_j in enumerate(panels):
            if i != j:
                A[i, j] = 0.5 / numpy.pi * integral(panel_i.xc, panel_i.yc, panel_j, numpy.cos(panel_i.beta),
                                                    numpy.sin(panel_i.beta))
    return A


########################################################################################################################
# DEFINING THE MATRIX FOR VORTEX, FROM THE FLOW TANGENCY CONDITION

def vortex_contribution_normal(panels):
    A = numpy.empty((panels.size, panels.size), dtype=float)
    # vortex contribution on a panel from itself
    numpy.fill_diagonal(A, 0.0)
    # vortex contribution on a panel from others
    for i, panel_i in enumerate(panels):
        for j, panel_j in enumerate(panels):
            if i != j:
                A[i, j] = -0.5 / numpy.pi * integral(panel_i.xc, panel_i.yc,
                                                     panel_j, numpy.sin(panel_i.beta), -numpy.cos(panel_i.beta))
    return A


########################################################################################################################
# STORING THE VALUES INTO DESIGNATED MATRICES

A_source = source_contribution_normal(panels)
B_vortex = vortex_contribution_normal(panels)


########################################################################################################################
# ENFORCING KUTTA CONDITION
# [(A11t + AN1t) ....... (A1Nt + ANNt) {SUM(j = 1 to N) (B1Nt + BNNt)}] * [SINGULARITIES] = -(b1t + bnt)
# Bn = At
# Bt = -An
# We will use above relations to compute
def kutta_condition(A_source, B_vortex):
    kutta = numpy.empty(A_source.shape[0] + 1, dtype=float)
    kutta[:-1] = B_vortex[0, :] + B_vortex[-1, :]  # At = Bn
    kutta[-1] = - numpy.sum(A_source[0, :] + A_source[-1, :])  # Bt = -An
    return kutta


########################################################################################################################
# BUILDING THE OVERALL N+1 X N+1 MATRIX

def build_singularity_matrix(A_source, B_vortex):
    A = numpy.empty((A_source.shape[0] + 1, A_source.shape[1] + 1), dtype=float)
    A[:-1, :-1] = A_source  # Source contribution array.... N x N
    A[:-1, -1] = numpy.sum(B_vortex, axis=1)  # Vortex contribution array..... N x 1
    A[-1, :] = kutta_condition(A_source, B_vortex)  # Kutta condition array.... 1 x N+1
    return A


########################################################################################################################
# CREATING THE RHS ARRAY

def build_freestream_rhs(panels, freestream):
    b = numpy.empty(panels.size + 1, dtype=float)

    # Freestream contribution on each panel
    for i, panel in enumerate(panels):
        b[i] = -freestream.u_inf * numpy.cos(freestream.alpha - panel.beta)

    # Freestream contribution on the Kutta condition
    b[-1] = -freestream.u_inf * (numpy.sin(freestream.alpha - panels[0].beta) +
                                 numpy.sin(freestream.alpha - panels[-1].beta))
    return b


########################################################################################################################
# STORING THE VALUES

A = build_singularity_matrix(A_source, B_vortex)
b = build_freestream_rhs(panels, freestream)
########################################################################################################################
# SOLVING FOR SIGMA AND GAMMA

strengths = numpy.linalg.solve(A, b)

# store source strength on each panel object
for i, panel in enumerate(panels):
    panel.sigma = strengths[i]

# store circulation density
gamma = strengths[-1]


########################################################################################################################
# CALCULATING TANGENTIAL VELOCITY
'''
def compute_tangential_velocity(panels, freestream, gamma, A_source, B_vortex):
    A = numpy.empty((panels.size, panels.size + 1), dtype=float)
    A[:, :-1] = B_vortex
    A[:, -1] = -numpy.sum(A_source, axis=1)

    # Freestream contribution
    b = freestream.u_inf * numpy.sin([freestream.alpha - panel.beta for panel in panels])

    strengths = numpy.append([panel.sigma for panel in panels], gamma)

    tangential_velocities = numpy.dot(A, strengths) + b

    for i, panel in enumerate(panels):
        panel.vt = tangential_velocities[i]


########################################################################################################################
# COMPUTING THE VALUES BY PASSING FUNCTION PARAMETERS

compute_tangential_velocity(panels, freestream, gamma, A_source, B_vortex)
########################################################################################################################
# CALCULATING Cp

for panel in panels:
    panel.cp = 1.0 - (panel.vt / freestream.u_inf)**2

'''
########################################################################################################################
# COMPUTING CARTESIAN VELOCITIES

def get_velocity_field(panels, freestream, X, Y):
    u = freestream.u_inf * math.cos(freestream.alpha) * numpy.ones_like(X, dtype=float)
    v = freestream.u_inf * math.sin(freestream.alpha) * numpy.ones_like(X, dtype=float)
    vec_integral = numpy.vectorize(integral)

    for panel in panels:
        u += panel.sigma / (2.0 * math.pi) * vec_integral(X, Y, panel, 1.0, 0.0) - gamma / (
                    2.0 * math.pi) * vec_integral(X, Y, panel, 0.0, 1.0)
        v += panel.sigma / (2.0 * math.pi) * vec_integral(X, Y, panel, 0.0, 1.0) - gamma / (
                    2.0 * math.pi) * vec_integral(X, Y, panel, 1.0, 0.0)

    return u, v


m = 20  # Number of the grid points

# Defining domain and grid
x_domain = numpy.linspace(-1.0, 2.0, m)
y_domain = numpy.linspace(-0.3, 0.3, m)
X, Y = numpy.meshgrid(x_domain, y_domain)
u, v = get_velocity_field(panels, freestream, X, Y)
########################################################################################################################
# SOME IMPORTANT CALCULATIONS

# Accuracy...closer to zero the better
accuracy = sum([panel.sigma * panel.length for panel in panels])
print(f"Accuracy = {accuracy}")

# Coefficient of lift
c = abs(max(panel.xa for panel in panels) - min(panel.xa for panel in panels))
cl = (gamma * sum(panel.length for panel in panels) / (0.5 * freestream.u_inf * c))
print(f"Cl = {cl}")
########################################################################################################################
# PLOTTING

# Plotting the streamlines
pyplot.xlim(-1.0, 2.0)
pyplot.ylim(-0.3, 0.3)
pyplot.plot(x, y)
pyplot.streamplot(X, Y, u, v, density=3, linewidth=1, color='black', arrowsize=1, arrowstyle='->')
pyplot.fill_between(x, y, color='k')

# Plotting Cp
"""
pyplot.plot([panel.xc for panel in panels if panel.loc == 'upper'],
            [panel.cp for panel in panels if panel.loc == 'upper'], label='upper surface', color='r', linestyle='-',
            linewidth=2, marker='o', markersize=4)
pyplot.plot([panel.xc for panel in panels if panel.loc == 'lower'],
            [panel.cp for panel in panels if panel.loc == 'lower'], label= 'lower surface', color='b',  linestyle='-',
            linewidth=1, marker='o', markersize=4)
pyplot.xlim(-0.1, 1.1)
pyplot.ylim(1.0, -2.0)
"""
pyplot.show()