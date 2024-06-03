"""
Created on Mon Jun  3 16:44:40 2024.

@author: Quirijn B. van Woerkom
Code that propagates the equations of motion for Spacecraft
Attitude Dynamics and Control assignment 1.
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


class SpacecraftAttitude:
    """
    Class that computes and propagates the equations of motion governing
    the attitude of a spacecraft with three-axis stabilised control and
    a state estimator.
    """

    def __init__(self,
                 controller,
                 estimator,
                 Td: np.ndarray[float] = np.array([1e-3, 1e-3, 1e-3]),
                 J_11: float = 2500,
                 J_22: float = 2300,
                 J_33: float = 3100,
                 h: float = 700e3):
        """
        Initialise the SpacecraftAttitude object.

        Parameters
        ----------
        controller : instance of Controller class
            Controller to use.
        estimator : instance of Estimator class
            Estimator to use.
        Td : np.ndarray[float]
            Disturbance torques in the body frame in N m.
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
        # Save the disturbance torques (as this was not explicitly given,
        # I have assumed these to be in the body-fixed frame)
        self.Td = Td
        # Save the controller and estimator
        self.controller = controller
        self.state_estimator = estimator

    def compute_EoM(self,
                    t: float,
                    state: np.ndarray[float]):
        """
        Compute the equations of motion for the spacecraft attitude.

        Parameters
        ----------
        t : float
            Time in seconds. In principle not necessary, but required for the
            function signature to match the requirements of
            sp.integrate.solve_ivp.
        state : np.ndarray[float]
            State of the spacecraft attitude as defined in the report,
            in rad and rad/s.

        Returns
        -------
        state_deriv : np.ndarray[float]
            Derivative of the state, in rad/s and rad/s^2.
        """
        # While in principle a more vectorised expression ought to be more
        # efficient, I will prefer to write code to match the equations
        # in the report so as to avoid errors.
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
        # Preallocate output array
        state_derivs = np.zeros((6,))
        # Compute the control torques
        Mu = self.control_torque(state)
        # Pre-compute the total torque
        M1_J11 = -3/4*n**2*(1+np.cos(2*th2))*np.sin(2*th1)*C1 + \
            Td[0]/J_11 + Mu[0]/J_11
        M2_J22 = 3/2*n**2*np.sin(2*th2)*np.cos(th1)*C2 + \
            Td[1]/J_22 + Mu[1]/J_22
        M3_J33 = 3/2*n**2*np.sin(2*th2)*np.sin(th1)*C3 + \
            Td[2]/J_33 + Mu[2]/J_33
        # Compute the state derivative
        state_derivs[0] = omg1 + np.sin(th1)*np.tan(th2)*omg2 + \
            np.cos(th1)*np.tan(th2)*omg3 + n*np.sin(th3)/np.cos(th2)
        state_derivs[1] = np.cos(th1)*omg2 - np.sin(th1)*omg3 + \
            n*np.cos(th3)
        state_derivs[2] = np.sin(th1)/np.cos(th2)*omg2 + \
            np.cos(th1)/np.cos(th2)*omg3 + n*np.tan(th2)*np.sin(th3)
        state_derivs[3] = C1*omg2*omg3 + M1_J11
        state_derivs[4] = C2*omg1*omg3 + M2_J22
        state_derivs[5] = C3*omg1*omg2 + M3_J33
        return state_derivs

    def control_torque(self,
                       state: np.ndarray[float]):
        """
        Determine the control torque for a given state.

        Parameters
        ----------
        state : np.ndarray[float]
            State for which to determine the control torque.

        Returns
        -------
        Mu : np.ndarray[float]
            Commanded control torque.
        """
        # Read the state from the sensors
        sensor_state = self.state_estimator.sensor(state)
        # Pass the state found by the sensor through the Kalman filter, and
        # obtain an estimate for the true state
        est_state = self.state_estimator.estimate(sensor_state)
        # Determine the control response from the controller for this state
        Mu = self.controller.response(est_state)
        return Mu

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
        Propagate the equations of motion over a given number of orbits.

        Propagate the equations of motion over a given number of orbits
        (not necessarily integral) given some initial state, and return the
        attitude history.

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
        integrator = sp.integrate.solve_ivp(self.compute_EoM,
                                            tint,
                                            initial_state,
                                            method=method,
                                            rtol=rtol,
                                            atol=atol)
        # Extract the time and state history
        t_arr = integrator.t
        state_hist = integrator.y
        return t_arr, state_hist
