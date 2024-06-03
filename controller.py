"""
Created on Mon Jun  3 17:42:31 2024.

@author: Quirijn B. van Woerkom
Code that models the controller for a spacecraft attitude control system.
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


class Controller:
    """
    Class that models the controller for a spacecraft attitude control system.

    Self-contained from the equations of motion class.
    """

    def __init__(self,
                 h: float = 700e3,
                 reference_attitude: np.ndarray[float] = np.array([0, 0, 0]),
                 gains: np.ndarray[float] = np.array([0, 0, 0, 0, 0, 0])):
        """
        Initialise the controller.

        Parameters
        ----------
        h : float
            Altitude of the spacecraft in m.
        reference_attitude : np.ndarray[float]
            Reference attitude of the controller, with angles given in rad.
        gains : np.ndarray[float]
            Gains for the controller, with an entry per state variable.
        """
        # Save the reference attitude and gains
        self.ref_att = reference_attitude
        self.gains = gains
        # Compute the mean motion, and save the mean motion and altitude
        n = np.sqrt(const.GM_earth.value/(const.R_earth.value+h)**3)
        self.n = n
        self.h = h

    def response(self,
                 est_state: np.ndarray[float]):
        """
        Compute the commanded response to a given estimated state.

        Parameters
        ----------
        est_state : np.ndarray[float]
            Estimated state.

        Returns
        -------
        Mu : np.ndarray[float]
            Commanded control torque.
        """
        att_err = est_state[:3] - self.ref_att
        est_rate = self.calc_Euler_rates(est_state)
        Mu = att_err*self.gains[:3] + est_rate*self.gains[3:]
        return Mu

    def calc_Euler_rates(self,
                         est_state: np.ndarray[float]):
        """
        Compute the rates of the Euler angles for use in the PD-controller.

        Compute an estimate for the rate of change of the Euler angles with
        respect to time based on the estimated state. This allows the
        PD-controller to determine the D-component of the commanded torque.

        Parameters
        ----------
        est_state : np.ndarray[float]
            Estimated state.

        Returns
        -------
        est_Euler_rates : np.ndarray[float]
            Estimated value of the Euler angle rates for this state.
        """
        # Write some aliases for the state quantities, for legibility
        th1 = est_state[0]  # rad, theta1
        th2 = est_state[1]  # rad, theta2
        th3 = est_state[2]  # rad, theta3
        omg1 = est_state[3]  # rad/s, omega1
        omg2 = est_state[4]  # rad/s, omega2
        omg3 = est_state[5]  # rad/s, omega3
        # Extract attributes
        n = self.n
        # Compute the derivative of the Euler angles
        est_Euler_rates = np.zeros((3,))
        est_Euler_rates[0] = omg1 + np.sin(th1)*np.tan(th2)*omg2 + \
            np.cos(th1)*np.tan(th2)*omg3 + n*np.sin(th3)/np.cos(th2)
        est_Euler_rates[1] = np.cos(th1)*omg2 - np.sin(th1)*omg3 + \
            n*np.cos(th3)
        est_Euler_rates[2] = np.sin(th1)/np.cos(th2)*omg2 + \
            np.cos(th1)/np.cos(th2)*omg3 + n*np.tan(th2)*np.sin(th3)
        return est_Euler_rates
