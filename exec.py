"""
Created on Mon Jun  3 18:37:18 2024.

@author: Quirijn B. van Woerkom
Execution file for SADC assignment 1.
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

# Import custom classes
from controller import Controller
from state_estimator import Estimator
from EoM import SpacecraftAttitude
from verification import LinearisedEoM

# Plotstyle changes
# Increase the matplotlib font size
plt.rcParams.update({"font.size": 22})
# Set layout to tight
plt.rcParams.update({"figure.autolayout": True})
# Set grid to true
plt.rcParams.update({"axes.grid": True})
# Set the plotting style to interactive (autoshows plots)
plt.ion()

# %% Initialise an idle controller and perfect estimator.
idle_controller = Controller(
    h=700e3, reference_attitude=np.zeros((3,)),
    gains=np.zeros((6,)))  # Never commands any torque
perfect_estimator = Estimator(
    att_std=np.zeros((3,)), gyro_bias=np.zeros((3,)))  # Gives perfect
# state estimates
# Use these to initialise a spacecraft attitude instance
attitude_perfect = SpacecraftAttitude(idle_controller, perfect_estimator)
# Also initialise a LinearisedEoM instance
linear_eom = LinearisedEoM()
# Propagate
initial_rates = np.array([1, 1, 1])*0
initial_att = np.array([1, 1, 1])
t_arr, state_hist = attitude_perfect.propagate_EoM(1/30,
                                                   initial_rates,
                                                   initial_att)
t_arr_l, state_hist_l = linear_eom.propagate_EoM(1/30,
                                                 initial_rates,
                                                 initial_att)

# Plot the results
fig, ax = plt.subplots(1, 2)
# Plot the Euler angles
# Full EoM
ax[0].plot(t_arr, state_hist[0, :]*180/np.pi, label='$\\theta_1$')
ax[0].plot(t_arr, state_hist[1, :]*180/np.pi, label='$\\theta_2$')
ax[0].plot(t_arr, state_hist[2, :]*180/np.pi, label='$\\theta_3$')
# Linearised EoM
ax[0].plot(t_arr_l, state_hist_l[0, :]*180/np.pi, label='$\\theta_1$, lin.')
ax[0].plot(t_arr_l, state_hist_l[1, :]*180/np.pi, label='$\\theta_2$, lin.')
ax[0].plot(t_arr_l, state_hist_l[2, :]*180/np.pi, label='$\\theta_3$, lin.')
ax[0].set_xlabel('Time [s]')
ax[0].set_ylabel('Euler angle [rad]')
ax[0].legend()
# Plot the rotational velocities
# Full EoM
ax[1].plot(t_arr, state_hist[3, :]*180/np.pi, label='$\\omega_1$')
ax[1].plot(t_arr, state_hist[4, :]*180/np.pi, label='$\\omega_2$')
ax[1].plot(t_arr, state_hist[5, :]*180/np.pi, label='$\\omega_3$')
# Linearised EoM
ax[1].plot(t_arr_l, state_hist_l[3, :]*180/np.pi, label='$\\omega_1$, lin.')
ax[1].plot(t_arr_l, state_hist_l[4, :]*180/np.pi, label='$\\omega_2$, lin.')
ax[1].plot(t_arr_l, state_hist_l[5, :]*180/np.pi, label='$\\omega_3$, lin.')
ax[1].set_xlabel('Time [s]')
ax[1].set_ylabel('Angular velocity [rad/s]')
ax[1].legend()
#
