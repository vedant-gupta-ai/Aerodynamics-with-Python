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
# STORING THE PANEL POINTS

airfoil_x = 'W Case Main_x.txt'
x_af = numpy.loadtxt(airfoil_x, dtype=float, unpack=True)  # Unpacking x coordinates of the main foil
airfoil_y = 'W Case Main_y.txt'
y_af = numpy.loadtxt(airfoil_y, dtype=float, unpack=True)  # Unpacking y coordinates of the main foil

flap_x = 'W Case Flap_x.txt'
x_f = numpy.loadtxt(flap_x, dtype=float, unpack=True)  # Unpacking x coordinates of the flap foil
flap_y = 'W Case Flap_y.txt'
y_f = numpy.loadtxt(flap_y, dtype=float, unpack=True)  # Unpacking y coordinates of the flap foil


########################################################################################################################
# DEFINING THE FUNCTION TO ROTATE THE AXIS SYSTEM FOR THE FLAP

def rotate(xc, yc, theta, x, y):
    # xc, yc -- coordinates of rotation center
    # x, y -- coordinates to rotate
    # theta -- degrees of flap rotation
    theta1 = theta / 180 * numpy.pi
    x_1, y_1 = 0, 0
    x_1 = x - xc * numpy.ones_like(x)
    y_1 = y - yc * numpy.ones_like(y)
    x_r = x_1 * numpy.cos(theta1) + y_1 * numpy.sin(theta1) + xc * numpy.ones_like(x)
    y_r = x_1 * (-1) * numpy.sin(theta1) + y_1 * numpy.cos(theta1) + yc * numpy.ones_like(y)
    return x_r, y_r


x_fn, y_fn = rotate(1.03, -0.054, 20.0, x_f, y_f )  # Storing the new flap points as x_fn, y_fn

########################################################################################################################
# CREATING THE PANEL OBJECTS FOR AIRFOIL AND FLAP

N = 150  # Number of panels
panels_af = numpy.empty(N, dtype=object)  # First creating the empty array
for i in range(N):
    panels_af[i] = Panel(x_af[i], y_af[i], x_af[i+1], y_af[i+1])

panels_flap = numpy.empty(N, dtype=object)
for i in range(N):
    panels_flap[i] = Panel(x_fn[i], y_fn[i], x_fn[i+1], y_fn[i+1])

panels = numpy.append(panels_af, panels_flap)


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
# FREESTREAM CONDITIONS

class Freestream:
    def __init__(self, alpha, u_inf):
        self.u_inf = u_inf
        self.alpha = numpy.radians(alpha)  # degrees to radians


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

def kutta_condition(A_source, B_vortex):
    b = numpy.empty([2, A_source.shape[1] + 2], dtype=float)
    b[0, :-2] = B_vortex[0, :] + B_vortex[N - 1, :]  # Effects by main
    b[1, :-2] = B_vortex[N, :] + B_vortex[2 * N - 1, :]  # Effects by flap
    b[0, -2] = - numpy.sum(A_source[0, :N] + A_source[N - 1, :N])  # Main on main
    b[0, -1] = - numpy.sum(A_source[0, N:] + A_source[N - 1, N:])  # Main on flap
    b[1, -2] = - numpy.sum(A_source[N, :N] + A_source[2 * N - 1, :N])  # Flap on main
    b[1, -1] = - numpy.sum(A_source[N, N:] + A_source[2 * N - 1, N:])  # Flap on flap
    return b


########################################################################################################################
# BUILDING THE OVERALL N+2 X N+2 MATRIX

def build_singularity_matrix(A_source, B_vortex):
    A = numpy.empty((A_source.shape[0]+2, A_source.shape[0]+2), dtype=float)
    A[:-2, :-2] = A_source
    A[:-2, -2] = numpy.sum(B_vortex[:, :N], axis=1)  # Effect on main
    A[:-2, -1] = numpy.sum(B_vortex[:, N:], axis=1)  # Effect on flap
    A[-2:, :] = kutta_condition(A_source, B_vortex)
    return A


########################################################################################################################
# DEFINING RHS TERM

def build_freestream_rhs(panels, freestream):
    b = numpy.empty(panels.size + 2, dtype=float)
    for i, panel in enumerate(panels):
        b[i] = -freestream.u_inf * numpy.cos(freestream.alpha - panel.beta)

    b[-2] = -freestream.u_inf*(numpy.sin(freestream.alpha-panels[0].beta) + numpy.sin(freestream.alpha-panels[N-1].beta))
    # Freestream effects by main foil
    b[-1] = -freestream.u_inf*(numpy.sin(freestream.alpha-panels[N].beta) + numpy.sin(freestream.alpha-panels[2*N-1].beta))
    # Freestream effects by flap
    return b


########################################################################################################################
# COMPUTING TANGENTIAL VELOCITY AND EVENTUALLY Cp

def compute_tangential_velocity(panels, freestream, gamma_flap, gamma_foil, A_source, B_vortex):
    A = numpy.empty((panels.size, panels.size + 2), dtype=float)
    A[:, :-2] = B_vortex
    A[:, -2] = -numpy.sum(A_source[:, :N], axis=1)  # Effects on main
    A[:, -1] = -numpy.sum(A_source[:, N:], axis=1)  # Effects on flap

    b = freestream.u_inf * numpy.sin([freestream.alpha-panel.beta for panel in panels])
    strengths = numpy.append([panel.sigma for panel in panels],gamma_flap)
    strengths = numpy.append(strengths, gamma_foil)
    tangential_velocities = numpy.dot(A, strengths) + b
    for i, panel in enumerate(panels):
        panel.vt = tangential_velocities[i]


########################################################################################################################
# COMPUTING LIFT

def compute_lift(panels, alpha):
    freestream = Freestream(alpha, u_inf=1.0)

    LHS = build_singularity_matrix(A_source, B_vortex)

    b = build_freestream_rhs(panels, freestream)

    strengths = numpy.linalg.solve(LHS, b)
    for i, panel in enumerate(panels):
        panel.sigma = strengths[i]
    gamma_flap = strengths[-2]
    gamma_foil = strengths[-1]

    compute_tangential_velocity(panels, freestream, gamma_flap, gamma_foil, A_source, B_vortex)

    for panel in panels:
        panel.cp = 1 - (panel.vt / freestream.u_inf) ** 2

    # MAIN GODDAMN FUNCTION TO COMPUTE LIFT!!!
    lift = 0
    for panel in panels:
        lift += -panel.cp * numpy.sin(panel.beta) * panel.length

    return lift


########################################################################################################################
# LOOP FOR BISECTION THEOREM

while True:
    alpha_a = -14.0
    alpha_b = 14.0
    alpha = (alpha_a + alpha_b) / 2

    if compute_lift(panels, alpha) == 0:
        alpha_zero = alpha
        print(f"Zero lift AoA = {alpha_zero}")
        break
    elif compute_lift(panels, alpha_a) * compute_lift(panels, alpha) < 0:
        alpha_b = alpha
    elif compute_lift(panels, alpha_b) * compute_lift(panels, alpha) < 0:
        alpha_a = alpha

########################################################################################################################
