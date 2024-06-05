"""
Created on Tue Jun  4 20:17:49 2024.

@author: Quirijn B. van Woerkom
Sympy code to find an expression for the differentiation gains.
"""
# Standard imports
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from astropy import constants as const
from astropy import units as u
from astropy.io import fits
import time
import tqdm

# Import sympy
import sympy as sym

# Plotstyle changes
# Increase the matplotlib font size
plt.rcParams.update({"font.size": 22})
# Set layout to tight
plt.rcParams.update({"figure.autolayout": True})
# Set grid to true
plt.rcParams.update({"axes.grid": True})
# Set the plotting style to interactive (autoshows plots)
plt.ion()


# Initialise pretty print
sym.init_printing()

# Define shorthands for the symbols we need
z1 = sym.symbols('zeta_1')  # Damping factor 1
z3 = sym.symbols('zeta_3')  # Damping factor 3
z = sym.symbols('zeta')  # Damping factor

omg1 = sym.symbols('omega_1')  # Natural frequency 1
omg3 = sym.symbols('omega_3')  # Natural frequency 3

A1 = sym.symbols('A_1')
A3 = sym.symbols('A_3')
B1 = sym.symbols('B_1')
B3 = sym.symbols('B_3')
C1 = sym.symbols('C_1')
C3 = sym.symbols('C_3')

# Define equations
eq1 = sym.Eq(omg1*omg1*omg3*omg3,
             B1*B3)
eq2 = sym.Eq(2*omg1*omg3*(z*omg3 + z*omg1),
             A1*B3 + A3*B1)
eq3 = sym.Eq(omg1**2 + omg3**2 + 4*z*z*omg1*omg3,
             B1 + B3 + A1*A3 + C1*C3)
eq4 = sym.Eq(2*(z*omg1 + z*omg3),
             A1 + A3)

# Solve them
eqs = [eq1, eq2, eq3, eq4]
sol = sym.solve(eqs, [A1, A3, omg1, omg3])
