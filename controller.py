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

# %% Auxiliary functions


def calc_kd(kp1: float,
            kp3: float,
            zeta: float,
            J_11: float = 2500,
            J_22: float = 2300,
            J_33: float = 3100,
            h: float = 700e3):
    """
    Compute the differentiation gains for the roll and yaw axes.

    Given the proportional gains and damping factor, compute the
    differentiation gains for the roll and yaw axes.

    Parameters
    ----------
    kp1 : float
        Proportional gain for the roll axis.
    kp3 : float
        Proportional gain for the yaw axis.
    zeta : float
        Damping factor for both axis.
    J_11 : float
        Moment of inertia about axis 1, in kg m^2.
    J_22 : float
        Moment of inertia about axis 2, in kg m^2.
    J_33 : float
        Moment of inertia about axis 3, in kg m^2.
    h : float
        Altitude of the spacecraft in m.

    Returns
    -------
    kd1 : float
        Differentiation gain for the roll axis.
    kd3 : float
        Differentiation gain for the yaw axis.
    """
    # Compute the mean motion
    n = np.sqrt(const.GM_earth.value/(const.R_earth.value+h)**3)
    # Compute the constants required for the computation
    B1 = (4*n**2*(J_22-J_33) - kp1)/J_11
    B3 = (n**2*(J_22-J_11) - kp3)/J_33
    C1 = n*(J_11-J_22+J_33)/J_33
    C3 = n*(J_11-J_22+J_33)/J_11
    K = -(1-np.sqrt(B3/B1))/(1-np.sqrt(B1/B3))
    # Compute the differentiation gains
    kd1 = -2*J_11*zeta*np.sqrt((B1+B3+C1*C3+2*(1-2*zeta**2)*np.sqrt(B1*B3)) /
                               (K**2+2*(1-2*zeta**2)*K + 1))
    kd3 = K*J_33/J_11*kd1
    return kd1, kd3


def settling_time(t_arr: np.ndarray[float],
                  signal: np.ndarray[float],
                  ref_val: float,
                  settling_val: float):
    """
    Compute the settling time for the given history.

    Parameters
    ----------
    t_arr : np.ndarray[float]
        Array of timestamps.
    signal : np.ndarray[float]
        Array of signal values corresponding to t_arr.
    ref_val : float
        Reference value.
    settling_val : float
        Value to within which ref_val must be approached.

    Returns
    -------
    t_sett : float
        Settling time.
    """
    settled_arr = np.where(np.abs(signal-ref_val) < settling_val,
                           True, False)
    # Loop through array backwards
    for idx, settled in enumerate(settled_arr[::-1]):
        # Check if not settled
        if not settled:
            # After encountering first value which has not settled,
            # save the settling time and break
            t_sett = t_arr[-idx+1]
            break
    return t_sett


# %% Class definition


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
            Entries must be given in Nm/rad and Nms/rad
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
