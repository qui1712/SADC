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
    """

    def __init__(self,
                 reference_attitude: np.ndarray[float] = None,
                 gains: np.ndarray[float] = None):
        """
        Initialise the controller.

        Parameters
        ----------
        reference_attitude : np.ndarray[float]
            Reference attitude of the controller, with angles given in rad.
        gains : np.ndarray[float]
            Gains for the controller.
        """
        self.ref_att = reference_attitude
        self.gains = gains

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
        return 0
