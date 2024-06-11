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

# Type annotation objects
from collections.abc import Callable

# Plotstyle changes
# Increase the matplotlib font size
plt.rcParams.update({"font.size": 22})
# Set layout to tight
plt.rcParams.update({"figure.autolayout": True})
# Set grid to true
plt.rcParams.update({"axes.grid": True})
# Set the plotting style to interactive (autoshows plots)
plt.ion()


class SimpleEstimator:
    """
    Estimator that straightforwardly takes the measurement as state estimate.
    """


class KalmanController:
    """
    Class modelling the state estimator for an attitude control system.

    Attributes
    ----------

    """

    def __init__(self,
                 controller,
                 att_std: np.ndarray[float] = np.array([.1, .1, .1]),
                 gyro_bias: np.ndarray[float] = np.array([.2, -.1, .15]),
                 Td: np.ndarray[float] = np.array([1e-3, 1e-3, 1e-3]),
                 x0: np.ndarray[float] = np.zeros((9,)),
                 P0: np.ndarray[float] = np.diag([10, 10, 10, .15, .15, .15,
                                                  .15, .15, .15])**2,
                 J_11: float = 2500,
                 J_22: float = 2300,
                 J_33: float = 3100,
                 h: float = 700e3,
                 method='RK45',
                 rtol=1e-7,
                 atol_att=1e-6,
                 atol_rate=1e-8,):
        """
        Initialise the estimator.

        Parameters
        ----------
        controller : instance of Controller
            An instance of the custom Controller class, which represents
            the controller of the spacecraft. Should have a method
            self.response which takes the state as input and returns the
            corresponding control torques as output.
        att_std : np.ndarray[float]
            Standard deviation of the white-normal noise attitude angle
            measurements in degrees.
        gyro_bias : np.ndarray[float]
            Bias of the angular velocity measurements in degree/s.
        Td : np.ndarray[float]
            Disturbance torques in the LVLH frame in Nm. Note that these are
            the standard deviations in each direction, not constant
            magnitudes.
        x0 : np.ndarray[float]
            Initial state estimate in rad and rad/s.
        P0 : np.ndarray[float]
            Initial state covariance matrix, with entries in degree^2 and
            degree^2/s^2.
        J_11 : float
            Moment of inertia about axis 1, in kg m^2.
        J_22 : float
            Moment of inertia about axis 2, in kg m^2.
        J_33 : float
            Moment of inertia about axis 3, in kg m^2.
        h : float
            Altitude of the spacecraft in m.
        method : str
            Method to pass to sp.integrate.solve_ivp.
        rtol : float
            Relative tolerance for integration of all state variables.
        atol_att : float
            Absolute tolerance for the attitudes in degrees.
        atol_rate : float
            Absolute tolerance for the angular velocities in deg/s.
        """
        # Save the controller as attribute
        self.controller = controller
        # Save the noise attributes as rad and rad/s
        self.att_std = att_std/180*np.pi
        self.gyro_bias = gyro_bias/180*np.pi
        # Save the integration settings
        self.method = method
        # Convert the absolute tolerance to rad and rad/s
        atol = np.ones((6,))
        atol[:3] *= atol_att/180*np.pi
        atol[3:] *= atol_rate/180*np.pi
        self.atol = atol
        self.rtol = rtol
        # Prepare some attributes to avoid unnecessary recomputation, and
        # initialise the filter
        # Save the matrices G, H, H^T, R, and Q
        self.G = np.diag([0, 0, 0, 1/J_11, 1/J_22, 1/J_33])
        self.Q = np.diag([0, 0, 0, (Td[0]/J_11)**2, (Td[1]/J_22)**2,
                          (Td[2]/J_33)**2])
        self.R = np.diag([att_std[0]**2, att_std[1]**2, att_std[2]**2,
                          0, 0, 0])
        self.H = np.hstack((np.eye(6),
                            np.vstack(np.zeros((3, 3)),
                                      -np.eye(3))))
        self.HT = self.H.T
        # Compute and save the mean motion, altitude and orbital period
        n = np.sqrt(const.GM_earth.value/(const.R_earth.value+h)**3)
        self.n = n
        self.h = h
        self.P = 2*np.pi/n
        # Save the initial state estimate and covariance estimate,
        # converted to rad
        self.x_kk = x0/(180*np.pi)
        self.P_kk = P0/(180*np.pi)**2

    def control_torque(self,
                       t: float,
                       state: np.ndarray[float]):
        """
        Determine the control torque for a given time and state.

        Parameters
        ----------
        time : float
            Time at which to determine the control torque.
        state : np.ndarray[float]
            State for which to determine the control torque.

        Returns
        -------
        Mu : np.ndarray[float]
            Commanded control torque.
        """
        # Determine the control response from the controller for this state
        Mu = self.controller.response(state)
        return Mu

    def update_predict(self,
                       measurement: np.ndarray[float],
                       state_pred: np.ndarray[float],
                       P_kk: np.ndarray[float],
                       K_k: np.ndarray[float]):
        """
        Perform the update and prediction-step.

        Parameters
        ----------
        measurement : np.ndarray[float]
            Measurement of the state.
        state_pred : np.ndarray[float]
            Prediction of the state from the previous state estimate.
        P_kk : np.ndarray[float]
            Estimated state covariance matrix.
        K_k : np.ndarray[float]
            Kalman gain.

        Returns
        -------
        u_k : function
            Scheduled control inputs until the next measurement time. Takes
            inputs (time, state) and output the control moments.
        P_k1k : np.ndarray[float]
            Predicted state covariance matrix.
        x_k1k : np.ndarray[float]
            Predicted state at the next measurement epoch.
        """
        ...
        return u_k, P_k1k, x_k1k

    def precompute(self,
                   P_k1k: np.ndarray[float],
                   x_k1k: np.ndarray[float]):
        """
        Precompute the covariance matrix and Kalman gain for the next epoch.

        Parameters
        ----------
        P_k1k : np.ndarray[float]
            Predicted state covariance matrix.
        x_k1k : np.ndarray[float]
            Predicted state at the next measurement epoch.

        Returns
        -------
        K_k1 : np.ndarray[float]
            Kalman gain for the next measurement epoch.
        P_k1k1 : np.ndarray[float]
            Covariance matrix for the state error at the next measurement
            epoch.
        """
        # Retrieve some matrices saved as attributes
        HT = self.HT
        H = self.H
        R = self.R
        # Kalman gain computation #############################################
        # First perform the inversion
        inv = sp.linalg.inv(H @ P_k1k @ HT + R)
        K_k1 = P_k1k @ HT @ inv
        # State estimation error covariance matrix computation ################
        P_k1k1 = (np.eye(self.x_kk.shape[0]) - K_k1 @ H) @ P_k1k
        return K_k1, P_k1k1

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
        measurement : np.ndarray[float]
            A (corrupted) measurement of the state.
        """
        # Draw the attitude offset from a centred normal distribution
        att_offset = np.random.normal(loc=0,
                                      scale=self.att_std,
                                      size=(3,))
        # The gyro bias is fixed, and no noise is added to that
        # Compute and return the corrupted state estimate
        measurement = state[:6]
        measurement[:3] = measurement[:3] + att_offset
        measurement[3:] = measurement[3:] + self.gyro_bias
        return measurement

    def estimate(self,
                 tk: float,
                 measurement_dt: float,
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
        tk : float,
            Time at which sensor_state was obtained in s.
        measurement_dt : float
            Time from tk until next sensor measurement in s.

        Returns
        -------
        est_state : np.ndarray[float]
            An estimate for the current state.
        pred_state : np.ndarray[float]
            A prediction for the state at the next measurement epoch.
        Mu : function
            Commanded control input over the timespan until the next
            measurement epoch, with call signature (t, state).
        """
        if self.perfect:
            # For a perfect estimator, do not pass through the whole Kalman
            # filter routine, and just return the sensor state.
            est_state = np.copy(sensor_state)
            return est_state
        else:
            # Do Kalman filter magic:
            # Measurement update ##############################################

            # Prediction and control scheduling ###############################
            tint = [tk, tk+measurement_dt]
            # Define the equations of motion with perfect control input

            def perfect_eom(t, state):
                return self.compute_EoM(t, state, self.control_torque)

            # Perform the prediction step #####################################
            integrator = sp.integrate.solve_ivp(perfect_eom,
                                                tint,
                                                self.x_kk,
                                                method=self.method,
                                                rtol=self.rtol,
                                                atol=self.atol)
            predict_state_hist = integrator.y
            predict_t_arr = integrator.t
            # Determine control input by interpolating between tk and tk1
            # Produce the history of control inputs at each t in predict_t_arr
            predict_u = np.zeros((3, predict_t_arr.shape[0]))
            for idx, t_p in enumerate(predict_t_arr):
                predict_u[:, idx] = self.control_torque(
                    t_p, predict_state_hist[:, idx])

            # Create an interpolant using cubic splines
            interpolant = sp.interpolate.CubicSpline(
                predict_t_arr,
                predict_u,
                axis=1)
            # Define the commanded control input as function of time

            def commanded_Mu(t, state):
                return interpolant(t)

            # Compute the state prediction error covariance matrix

            # Compute the Kalman gain

            ###################################################################
            est_state = ...
            return est_state


class SpacecraftAttitude:
    """
    Class to compute and propagate the attitude equations of motion.

    Class that computes and propagates the equations of motion governing
    the attitude of a spacecraft with three-axis stabilised control and
    a state estimator on a circular orbit.
    """

    def __init__(self,
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
        estimator : instance of Estimator class
            Estimator to use.
        Td : np.ndarray[float]
            Disturbance torques in the LVLH frame in Nm.
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
        # Save the estimator as attribute
        self.state_estimator = estimator

    def compute_EoM(self,
                    t: float,
                    state: np.ndarray[float],
                    Mu: Callable):
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
        Mu : function
            Commanded control torques as function of time and state. Must
            have the signature Mu(t, state).

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
        # Calculate the control moment
        Mu_comp = Mu(t, state)
        # Pre-compute the total torque
        M1_J11 = -3/4*n**2*(1+np.cos(2*th2))*np.sin(2*th1)*C1 + \
            Td[0]/J_11 + Mu_comp[0]/J_11
        M2_J22 = 3/2*n**2*np.sin(2*th2)*np.cos(th1)*C2 + \
            Td[1]/J_22 + Mu_comp[1]/J_22
        M3_J33 = 3/2*n**2*np.sin(2*th2)*np.sin(th1)*C3 + \
            Td[2]/J_33 + Mu_comp[2]/J_33
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

    def propagate_EoM(self,
                      t0: float,
                      tend: float,
                      measurement_dt: float,
                      initial_rates: np.ndarray[float],
                      initial_att: np.ndarray[float] = np.array([10,
                                                                 10,
                                                                 10]),
                      rtol=1e-7,
                      atol_att=1e-6,
                      atol_rate=1e-8,
                      method='RK45'):
        """
        Propagate the equations of motion over a given timespan.

        Propagate the equations of motion over a given number of timespan
        given some initial state, and return the attitude history.

        Parameters
        ----------
        initial_rates : np.ndarray[float]
            Initial rates in deg/s.
        initial_att : np.ndarray[float]
            Initial attitudes in degrees.
        t0 : float
            Starting time.
        tend : float
            Final time.
        measurement_dt : float
            Time between sensor measurements in s.
        method : str
            Method to pass to sp.integrate.solve_ivp.
        rtol : float
            Relative tolerance of all state variables. Used in both the
            propagation of the true state and the Kalman filter.
        atol_att : float
            Absolute tolerance for the attitudes in degrees. Used in both the
            propagation of the true state and the Kalman filter.
        atol_rate : float
            Absolute tolerance for the angular velocities in deg/s. Used in
            both the propagation of the true state and the Kalman filter.

        Returns
        -------
        t_arr_tot : np.ndarray[float]
            Timestamps for the state history in seconds.
        state_hist_tot : np.ndarray[float]
            True attitude state history of the spacecraft.
        """
        # Convert the true initial state to rad and rad/s
        initial_state = np.zeros((6,))
        initial_state[:3] = initial_att/180*np.pi
        initial_state[3:] = initial_rates/180*np.pi
        # Convert the absolute tolerance to rad and rad/s
        atol = np.ones((6,))
        atol[:3] *= atol_att/180*np.pi
        atol[3:] *= atol_rate/180*np.pi
        # Propagate the equations of motion between measurements, take the
        # measurement, and determine the control input over the next
        # interval by propagating the state estimate
        # Initialise the loop
        t_arr_tot = np.copy(np.array([t0]))
        state_hist_tot = np.copy(initial_state)
        commanded_Mu_hist = np.zeros((3,))
        true_state_k = np.copy(initial_state)
        tk = t0
        tk1 = tk + measurement_dt
        # Enter the loop
        while tk1 < tend:
            # State estimator #################################################
            # Take a measurement of the true state
            z_k = self.estimator.sensor(true_state_k)
            # Feed this measurement into the state estimator to compute
            # an estimate for the state at tk, a prediction for the state
            # at tk1, and a prescribed control moment over [tk, tk1]
            x_kk, x_k1k, Mu = self.estimator.estimate(z_k)
            ###################################################################

            # Propagate to next measurement epoch #############################
            # Set the integration interval
            tint = [tk, tk1]
            # Propagation of the true state
            # Define the equations of motion with the commanded control input

            def command_eom(t, state):
                return self.compute_EoM(t, state, Mu)

            # Perform the integration
            integrator = sp.integrate.solve_ivp(command_eom,
                                                tint,
                                                initial_state,
                                                method=method,
                                                rtol=rtol,
                                                atol=atol)
            # Extract the time and state history
            t_arr = integrator.t
            state_hist = integrator.y
            # Append the state and time, excepting the first (which was the
            # final value for the previous integration interval)
            t_arr_tot = np.hstack((t_arr_tot, t_arr[1:]))
            state_hist_tot = np.hstack((state_hist_tot, state_hist[:, 1:]))
            # Compute the commanded control input at each time
            command_u = np.zeros((3, t_arr.shape[0]))
            for idx, t in enumerate(t_arr):
                command_u[:, idx] = Mu(t, state_hist[:, idx])
            # Append it to the list of commanded control inputs
            commanded_Mu_hist = np.hstack((commanded_Mu_hist, command_u))
            # Save the true state at the measurement epoch
            true_state_k = state_hist_tot[:, -1]
            ###################################################################

            # # Set t0, tf, the initial state and the corrected estimated state
            # # for the next loop
            # tk = t_arr_tot[-1]
            # tk1 = tk + measurement_dt
            # initial_state = state_hist_tot[:, -1]

        return t_arr, state_hist
