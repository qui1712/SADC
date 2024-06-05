"""
Created on Mon Jun  3 20:04:55 2024.

@author: Quirijn B. van Woerkom
Verification code for SADC assignment 1
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


# Plotstyle changes
# Increase the matplotlib font size
plt.rcParams.update({"font.size": 22})
# Set layout to tight
plt.rcParams.update({"figure.autolayout": True})
# Set grid to true
plt.rcParams.update({"axes.grid": True})
# Set the plotting style to interactive (autoshows plots)
plt.ion()


class LinearisedEoM:

    def __init__(self,
                 J_11: float = 2500,
                 J_22: float = 2300,
                 J_33: float = 3100,
                 h: float = 700e3,
                 Td: np.ndarray[float] = np.array([1e-3, 1e-3, 1e-3])):
        """
        Initialise the linearised EoM.

        Parameters
        ----------
        J_11 : float
            Moment of inertia about axis 1, in kg m^2.
        J_22 : float
            Moment of inertia about axis 2, in kg m^2.
        J_33 : float
            Moment of inertia about axis 3, in kg m^2.
        h : float
            Altitude of the spacecraft in m.
        """
        # Save the moments of inertia to attributes
        self.J_11 = J_11
        self.J_22 = J_22
        self.J_33 = J_33
        # Save often-occurring derivative quantities to avoid recomputation
        self.C1 = (J_22-J_33)/J_11
        self.C2 = (J_33-J_11)/J_22
        self.C3 = (J_11-J_22)/J_33
        # Compute the mean motion, and save the mean motion, altitude and
        # orbital period
        n = np.sqrt(const.GM_earth.value/(const.R_earth.value+h)**3)
        self.n = n
        self.h = h
        self.P = 2*np.pi/n
        # Save the disturbance torque
        self.Td = Td

    def calc_EoM(self,
                 t: float,
                 state: np.ndarray[float]):
        """
        Compute the EoM using the linearised formula.

        Compute the linearised equations of motion: do note that these are
        only valid for small Euler angles.

        Parameters
        ----------
        t : float
            Time.
        state : np.ndarray[float]
            State, with the Euler angles in the first three rows and the
            rotational velocities in the last three.

        Returns
        -------
        state_deriv : np.ndarray[float]
            State derivative.
        """
        # Write some aliases for the state quantities, for legibility
        th1 = state[0]  # rad, theta1
        th2 = state[1]  # rad, theta2
        th3 = state[2]  # rad, theta3
        omg1 = state[3]  # rad/s, omega1
        omg2 = state[4]  # rad/s, omega2
        omg3 = state[5]  # rad/s, omega3
        # Extract attributes
        n = self.n
        C1 = self.C1
        C2 = self.C2
        C3 = self.C3
        J_11 = self.J_11
        J_22 = self.J_22
        J_33 = self.J_33
        Td = self.Td
        # Compute the state derivative
        state_deriv = np.zeros((6,))
        # Commented out terms appear in manual linearisation of these
        # equations, but not in the expressions in the slides; turns out they
        # are nearly negligible :)
        state_deriv[0] = omg1 + n*th3  # + omg3*th2
        state_deriv[1] = omg2 + n  # - th1*omg3
        state_deriv[2] = omg3 + omg2*th1  # + th1*omg2  # + n*th2*th3
        state_deriv[3] = C1*omg3*omg2 - 3*n**2*th1*C1 + Td[0]/J_11
        state_deriv[4] = 3*n**2*C2*th2 + Td[1]/J_22  # + C2*omg1*omg3
        state_deriv[5] = C3*omg2*omg1 + Td[2]/J_33
        return state_deriv

    def propagate_EoM(self,
                      n_orbits: float,
                      initial_rates: np.ndarray[float],
                      initial_att: np.ndarray[float] = np.array([10,
                                                                 10,
                                                                 10]),
                      rtol=1e-7,
                      atol_att=1e-6,
                      atol_rate=1e-8,
                      method='RK45'):
        """
        Propagate the linearised EoM over a given number of orbits.

        Propagate the linearised equations of motion over a given number of
        orbits (not necessarily integral) given some initial state, and
        return the attitude history.

        Parameters
        ----------
        initial_rates : np.ndarray[float]
            Initial rates in deg/s.
        initial_att : np.ndarray[float]
            Initial attitudes in degrees.
        n_orbits : float
            Number of orbits over which to integrate. Need not be an integer.
        method : str
            Method to pass to sp.integrate.solve_ivp.
        rtol : float
            Relative tolerance of all state variables.
        atol_att : float
            Absolute tolerance for the attitudes in degrees.
        atol_rate : float
            Absolute tolerance for the angular velocities in deg/s.

        Returns
        -------
        t_arr : np.ndarray[float]
            Timestamps for the state history in seconds.
        state_hist : np.ndarray[float]
            Attitude state history of the spacecraft.
        """
        # Convert the initial state to rad and rad/s
        initial_state = np.zeros((6,))
        initial_state[:3] = initial_att/180*np.pi
        initial_state[3:] = initial_rates/180*np.pi
        # Convert the absolute tolerance to rad and rad/s
        atol = np.ones((6,))
        atol[:3] *= atol_att/180*np.pi
        atol[3:] *= atol_rate/180*np.pi

        # Set the integration interval
        t0 = 0
        tend = n_orbits*self.P
        tint = [t0, tend]
        # Perform the integration
        integrator = sp.integrate.solve_ivp(self.calc_EoM,
                                            tint,
                                            initial_state,
                                            method=method,
                                            rtol=rtol,
                                            atol=atol)
        # Extract the time and state history
        t_arr = integrator.t
        state_hist = integrator.y
        return t_arr, state_hist
