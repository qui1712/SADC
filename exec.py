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
from controller import Controller, calc_kd, settling_time
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

# %% Constants for quick reference
# Inertia properties:
J_11 = 2500  # kg m^2
J_22 = 2300  # kg m^2
J_33 = 3100  # kg m^2
Md = .001  # Nm, disturbance torque
delta = .1/180*np.pi  # rad, tracking error
# Orbital properties
h = 700e3  # m, altitude
n = np.sqrt(const.GM_earth.value/(
    const.R_earth.value+700e3)**3)  # rad/s, mean motion
P = 2*np.pi/n  # s, orbital period
measurement_rate = 100  # 1/s, measurement rate
# %% Verification of the equations of motion
# Initialise an idle controller and perfect estimator.
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
initial_rates = np.array([1, 1, 1])*1e-3
initial_att = np.array([2, 2, 2])
t_arr, state_hist = attitude_perfect.propagate_EoM(0,
                                                   1/50*P,
                                                   measurement_rate,
                                                   initial_rates,
                                                   initial_att)
t_arr_l, state_hist_l = linear_eom.propagate_EoM(0,
                                                 1/50*P,
                                                 initial_rates,
                                                 initial_att)

# Plot the results
fig, ax = plt.subplots(1, 2)
# Plot the Euler angles
# Full EoM
ax[0].plot(t_arr, state_hist[0, :]*180/np.pi, label='$\\theta_1$',
           linewidth=4)
ax[0].plot(t_arr, state_hist[1, :]*180/np.pi, label='$\\theta_2$',
           linewidth=4)
ax[0].plot(t_arr, state_hist[2, :]*180/np.pi, label='$\\theta_3$',
           linewidth=4)
# Linearised EoM
ax[0].plot(t_arr_l, state_hist_l[0, :]*180/np.pi, label='$\\theta_1$, lin.',
           linestyle='dotted', linewidth=4)
ax[0].plot(t_arr_l, state_hist_l[1, :]*180/np.pi, label='$\\theta_2$, lin.',
           linestyle='dotted', linewidth=4)
ax[0].plot(t_arr_l, state_hist_l[2, :]*180/np.pi, label='$\\theta_3$, lin.',
           linestyle='dotted', linewidth=4)
ax[0].set_xlabel('Time [s]')
ax[0].set_ylabel('Euler angle [$^{\\circ}$]')
ax[0].legend()
# Plot the rotational velocities
# Full EoM
ax[1].plot(t_arr, state_hist[3, :]*180/np.pi, label='$\\omega_1$',
           linewidth=4)
ax[1].plot(t_arr, state_hist[4, :]*180/np.pi, label='$\\omega_2$',
           linewidth=4)
ax[1].plot(t_arr, state_hist[5, :]*180/np.pi, label='$\\omega_3$',
           linewidth=4)
# Linearised EoM
ax[1].plot(t_arr_l, state_hist_l[3, :]*180/np.pi, label='$\\omega_1$, lin.',
           linestyle='dotted', linewidth=4)
ax[1].plot(t_arr_l, state_hist_l[4, :]*180/np.pi, label='$\\omega_2$, lin.',
           linestyle='dotted', linewidth=4)
ax[1].plot(t_arr_l, state_hist_l[5, :]*180/np.pi, label='$\\omega_3$, lin.',
           linestyle='dotted', linewidth=4)
ax[1].set_xlabel('Time [s]')
ax[1].set_ylabel('Angular velocity [$^{\\circ}$/s]')
ax[1].legend()
#

# %% Controller design
# Pitch controller
safety = 1.1
kp2 = (3*n**2*(J_11-J_33) - Md/delta)*safety
kd2_coeff = -2*np.sqrt(-kp2*J_22)
zeta = np.sqrt(2)/2  # Can be set freely.
kd2 = zeta*kd2_coeff
# Yaw and roll controller
kp1 = (4*n**2*(J_22-J_33) - Md/delta)*safety
kp3 = (n**2*(J_22-J_11) - Md/delta)*safety
kd1, kd3 = calc_kd(kp1, kp3, zeta)
# Initialise the controller with these settings
gains = np.array([kp1, kp2, kp3, kd1, kd2, kd3])
controller = Controller(gains=gains)
controlled_attitude = SpacecraftAttitude(controller, perfect_estimator)
# Propagate an example
# Propagate
initial_rates = np.zeros((3,))
initial_rates[1] = -n/np.pi*180
initial_rates += np.array([1, 1, 1])
initial_att = np.array([10, 10, 10])
t_arr, state_hist = controlled_attitude.propagate_EoM(0,
                                                      P,
                                                      measurement_rate,
                                                      initial_rates,
                                                      initial_att)
fig2, ax2 = plt.subplots(1, 2)
ax2[0].plot(t_arr, state_hist[0, :]*180/np.pi)
ax2[0].plot(t_arr, state_hist[1, :]*180/np.pi)
ax2[0].plot(t_arr, state_hist[2, :]*180/np.pi)

ax2[1].plot(t_arr, state_hist[3, :]*180/np.pi)
ax2[1].plot(t_arr, state_hist[4, :]*180/np.pi)
ax2[1].plot(t_arr, state_hist[5, :]*180/np.pi)
# %% Controller verification
# Nominal scenario: satellite starts at rest
# Reference scenario
initial_rates = np.zeros((3,))
initial_rates[1] = -n/np.pi*180
initial_att = np.array([10, 10, 10])
t_arr_ref, state_hist_ref = controlled_attitude.propagate_EoM(0,
                                                              .2*P,
                                                              measurement_rate,
                                                              initial_rates,
                                                              initial_att)
t_sett_ref1 = settling_time(t_arr_ref, state_hist_ref[0, :],
                            0, .1/180*np.pi)
t_sett_ref2 = settling_time(t_arr_ref, state_hist_ref[1, :],
                            0, .1/180*np.pi)
t_sett_ref3 = settling_time(t_arr_ref, state_hist_ref[2, :],
                            0, .1/180*np.pi)
# Plot the result
color1 = 'tab:blue'
color2 = 'tab:orange'
color3 = 'tab:green'
fig3, ax3 = plt.subplots(1, 2)
ax3[0].plot(t_arr_ref, state_hist_ref[0, :]*180/np.pi,
            color=color1, label='$\\theta_1$')
ax3[0].plot(t_arr_ref, state_hist_ref[1, :]*180/np.pi,
            color=color2, label='$\\theta_2$')
ax3[0].plot(t_arr_ref, state_hist_ref[2, :]*180/np.pi,
            color=color3, label='$\\theta_3$')
ax3[0].set_xlabel('Time [s]')
ax3[0].set_ylabel('Euler angle [$^{\\circ}$]')
ax3[0].set_yscale('symlog')

ax3[1].plot(t_arr_ref, state_hist_ref[3, :]*180/np.pi,
            color=color1, label='$\\omega_1$')
ax3[1].plot(t_arr_ref, state_hist_ref[4, :]*180/np.pi,
            color=color2, label='$\\omega_1$')
ax3[1].plot(t_arr_ref, state_hist_ref[5, :]*180/np.pi,
            color=color3, label='$\\omega_1$')
ax3[1].set_xlabel('Time [s]')
ax3[1].set_ylabel('Rotational velocity [$^{\\circ}/s$]')
# ax3[1].set_yscale('symlog')
# Perform the Monte Carlo analysis
n_draws = 100
t_sett = np.zeros((3, n_draws+1))
t_sett[0, 0] = t_sett_ref1
t_sett[1, 0] = t_sett_ref2
t_sett[2, 0] = t_sett_ref3
# Seed the random number generator
np.random.seed(4313)
for n_draw in range(n_draws):
    # Start at rest
    initial_rates = np.zeros((3,))
    initial_rates[1] = -n/np.pi*180
    # Draw random initial attitude in each direction
    initial_att = np.random.normal(0, scale=10, size=3)
    t_arr, state_hist = controlled_attitude.propagate_EoM(
        0, .2*P, measurement_rate, initial_rates, initial_att)
    t_sett[0, n_draw+1] = settling_time(t_arr, state_hist[0, :],
                                        0, .1/180*np.pi)
    t_sett[1, n_draw+1] = settling_time(t_arr, state_hist[1, :],
                                        0, .1/180*np.pi)
    t_sett[2, n_draw+1] = settling_time(t_arr, state_hist[2, :],
                                        0, .1/180*np.pi)
    # Plot the result
    alpha = .05
    lstyle = 'solid'
    ax3[0].plot(t_arr, state_hist[0, :]*180/np.pi,
                color=color1, alpha=alpha, linestyle=lstyle)
    ax3[0].plot(t_arr, state_hist[1, :]*180/np.pi,
                color=color2, alpha=alpha, linestyle=lstyle)
    ax3[0].plot(t_arr, state_hist[2, :]*180/np.pi,
                color=color3, alpha=alpha, linestyle=lstyle)

    ax3[1].plot(t_arr, state_hist[3, :]*180/np.pi,
                color=color1, alpha=alpha, linestyle=lstyle)
    ax3[1].plot(t_arr, state_hist[4, :]*180/np.pi,
                color=color2, alpha=alpha, linestyle=lstyle)
    ax3[1].plot(t_arr, state_hist[5, :]*180/np.pi,
                color=color3, alpha=alpha, linestyle=lstyle)
ax3[0].axhline(.1, color='grey', alpha=1, label='Accuracy\nrequirement')
ax3[1].axhline(-n/np.pi*180, color='grey', alpha=1, label='-n')
ax3[0].legend()
ax3[1].legend()
# %% Now run the perturbed scenario
# Reference scenario
initial_rates = np.zeros((3,))
initial_rates[1] = -n/np.pi*180
initial_rates += np.ones((3,))
initial_att = np.array([10, 10, 10])
t_arr_ref, state_hist_ref = controlled_attitude.propagate_EoM(0,
                                                              .2*P,
                                                              measurement_rate,
                                                              initial_rates,
                                                              initial_att)
t_sett_ref1 = settling_time(t_arr_ref, state_hist_ref[0, :],
                            0, .1/180*np.pi)
t_sett_ref2 = settling_time(t_arr_ref, state_hist_ref[1, :],
                            0, .1/180*np.pi)
t_sett_ref3 = settling_time(t_arr_ref, state_hist_ref[2, :],
                            0, .1/180*np.pi)
# Plot the result
color1 = 'tab:blue'
color2 = 'tab:orange'
color3 = 'tab:green'
fig4, ax4 = plt.subplots(1, 2)
ax4[0].plot(t_arr_ref, state_hist_ref[0, :]*180/np.pi,
            color=color1, label='$\\theta_1$')
ax4[0].plot(t_arr_ref, state_hist_ref[1, :]*180/np.pi,
            color=color2, label='$\\theta_2$')
ax4[0].plot(t_arr_ref, state_hist_ref[2, :]*180/np.pi,
            color=color3, label='$\\theta_3$')
ax4[0].set_xlabel('Time [s]')
ax4[0].set_ylabel('Euler angle [$^{\\circ}$]')
ax4[0].set_yscale('symlog')

ax4[1].plot(t_arr_ref, state_hist_ref[3, :]*180/np.pi,
            color=color1, label='$\\omega_1$')
ax4[1].plot(t_arr_ref, state_hist_ref[4, :]*180/np.pi,
            color=color2, label='$\\omega_1$')
ax4[1].plot(t_arr_ref, state_hist_ref[5, :]*180/np.pi,
            color=color3, label='$\\omega_1$')
ax4[1].set_xlabel('Time [s]')
ax4[1].set_ylabel('Rotational velocity [$^{\\circ}/s$]')
# ax4[1].set_yscale('symlog')
# Perform the Monte Carlo analysis
n_draws = 100
t_sett = np.zeros((3, n_draws+1))
t_sett[0, 0] = t_sett_ref1
t_sett[1, 0] = t_sett_ref2
t_sett[2, 0] = t_sett_ref3
# Seed the random number generator
np.random.seed(43132)
for n_draw in range(n_draws):
    # Start at rest
    initial_rates = np.zeros((3,))
    initial_rates[1] = -n/np.pi*180
    initial_rates += np.random.normal(0, scale=1, size=3)
    # Draw random initial attitude in each direction
    initial_att = np.random.normal(0, scale=10, size=3)
    t_arr, state_hist = controlled_attitude.propagate_EoM(
        0, .2*P, measurement_rate, initial_rates, initial_att)
    t_sett[0, n_draw+1] = settling_time(t_arr, state_hist[0, :],
                                        0, .1/180*np.pi)
    t_sett[1, n_draw+1] = settling_time(t_arr, state_hist[1, :],
                                        0, .1/180*np.pi)
    t_sett[2, n_draw+1] = settling_time(t_arr, state_hist[2, :],
                                        0, .1/180*np.pi)
    # Plot the result
    alpha = .05
    lstyle = 'solid'
    ax4[0].plot(t_arr, state_hist[0, :]*180/np.pi,
                color=color1, alpha=alpha, linestyle=lstyle)
    ax4[0].plot(t_arr, state_hist[1, :]*180/np.pi,
                color=color2, alpha=alpha, linestyle=lstyle)
    ax4[0].plot(t_arr, state_hist[2, :]*180/np.pi,
                color=color3, alpha=alpha, linestyle=lstyle)

    ax4[1].plot(t_arr, state_hist[3, :]*180/np.pi,
                color=color1, alpha=alpha, linestyle=lstyle)
    ax4[1].plot(t_arr, state_hist[4, :]*180/np.pi,
                color=color2, alpha=alpha, linestyle=lstyle)
    ax4[1].plot(t_arr, state_hist[5, :]*180/np.pi,
                color=color3, alpha=alpha, linestyle=lstyle)
ax4[0].axhline(.1, color='grey', alpha=1, label='Accuracy\nrequirement')
ax4[1].axhline(-n/np.pi*180, color='grey', alpha=1, label='-n')
ax4[0].legend()
ax4[1].legend()
