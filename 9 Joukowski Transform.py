import numpy as np
from matplotlib import pyplot
import math

c = 1
sections = 100
x_c = 0.0
y_c = 0.0
radius = 1.2
theta = np.linspace(0.0, 2*math.pi, sections)
x = np.cos(theta) + x_c
y = np.sin(theta) + y_c

z = radius * (x + y*1j)  # z-function

def zhu(z):
    zeta = z + c**2/z
    zeta_real = zeta.real
    zeta_imag = zeta.imag
    return zeta_real, zeta_imag


zeta_real, zeta_imag = zhu(z)

#Plotting
pyplot.title("Zeta Plane")
pyplot.xlim(-5.0, 5.0)
pyplot.ylim(-5.0, 5.0)
pyplot.plot(zeta_real, zeta_imag)
pyplot.show()
