"""
Created on Mon Jun  3 17:43:07 2024.

@author: Quirijn B. van Woerkom
Code that models the state estimator for a spacecraft attitude
control system.
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


class Estimator:
    """
    Class that models the state estimator for a spacecraft attitude control
    system.
    """

    def __init__(self,
                 att_std: np.ndarray[float] = np.array([.1, .1, .1]),
                 gyro_bias: np.ndarray[float] = np.array([.2, -.1, .15]),
                 seed=None):
        """
        Initialise the estimator.

        Parameters
        ----------
        att_std : np.ndarray[float]
            Standard deviation of the white-normal noise attitude angle
            measurements in degrees.
        gyro_bias : np.ndarray[float]
            Bias of the angular velocity measurements in degree/s.
        seed :
            Seed to seed the random number generator with. If None,
            the numbers become unreproducible.
        """
        # Seed the RNG
        np.random.seed(seed)
        # Save the attributes as rad and rad/s
        self.att_std = att_std/180*np.pi
        self.gyro_bias = gyro_bias/180*np.pi
        # If the bias and standard deviation are zero, mark the estimator as
        # a perfect estimator
        if np.all(att_std == 0) and np.all(gyro_bias == 0):
            self.perfect = True
        else:
            self.perfect = False

    def sensor(self,
               state: np.ndarray[float]):
        """
        Retrieve the state from the "sensors".

        Takes the given state and produces an imperfect estimate for that
        state using the given attitude control inaccuracy and gyro bias.

        Parameters
        ----------
        state : np.ndarray[float]
            (True) state at which to estimate the state.

        Returns
        -------
        sensor_state : np.ndarray[float]
            A (corrupted) estimate for the state.
        """
        # Draw the attitude offset from a centred normal distribution
        att_offset = np.random.normal(loc=0,
                                      scale=self.att_std,
                                      size=(3,))
        # The gyro bias is fixed, and no noise is added to that
        # Compute and return the corrupted state estimate
        sensor_state = np.copy(state)
        sensor_state[:3] = sensor_state[:3] + att_offset
        sensor_state[3:] = sensor_state[3:] + self.gyro_bias
        return sensor_state

    def estimate(self,
                 sensor_state: np.ndarray[float]):
        """
        Compute an estimate for the state.

        Compute an estimate for the state by first retrieving the sensor
        output and then passing this through an Extended Kalman Filter (EKF)
        to yield an estimate for the true state.

        Parameters
        ----------
        sensor_state : np.ndarray[float]
            A state retrieved from the sensors.

        Returns
        -------
        est_state : np.ndarray[float]
            A state estimate found from the EKF.
        """
        if self.perfect:
            # For a perfect estimator, do not pass through the whole Kalman
            # filter routine
            est_state = np.copy(sensor_state)
            return est_state
        else:
            # Do Kalman filter magic
            est_state = ...
            return est_state
