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
from state_estimator import (SimpleEstimator, PerfectEstimator,
                             KalmanEstimator,
                             SpacecraftAttitude,
                             Sensor)
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

# Constants for quick reference
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
measurement_dt = 1  # s, time between measurements
# %% Verification of the equations of motion
# Initialise an idle controller and no estimator.
idle_controller = None
no_estimator = None
# Use these to initialise a spacecraft attitude instance
attitude_perfect = SpacecraftAttitude(Td_mean=1e-3*np.ones((3,)),
                                      Td_std=np.zeros((3,)))
# Also initialise a LinearisedEoM instance
linear_eom = LinearisedEoM()
# Propagate
initial_rates = np.array([1, 1, 1])*1e-3
initial_att = np.array([2, 2, 2])
t_arr, state_hist = attitude_perfect.propagate_EoM(0,
                                                   1/50*P,
                                                   .1,
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
# %%
# Initialise a perfect estimator
perfect_estimator = PerfectEstimator(controller)
controlled_attitude = SpacecraftAttitude(estimator=perfect_estimator,
                                         Td_std=np.zeros((3,)),
                                         Td_mean=1e-3*np.ones((3,)))
# Controller verification
# Nominal scenario: satellite starts at rest
# Reference scenario
initial_rates = np.zeros((3,))
initial_rates[1] = -n/np.pi*180
initial_att = np.array([10, 10, 10])
t_arr_ref, state_hist_ref = controlled_attitude.propagate_EoM(0,
                                                              .2*P,
                                                              100,
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
            color=color2, label='$\\omega_2$')
ax3[1].plot(t_arr_ref, state_hist_ref[5, :]*180/np.pi,
            color=color3, label='$\\omega_3$')
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
        0, .2*P, 100, initial_rates, initial_att)
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
ax3[0].axhline(-.1, color='grey', alpha=1)
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
                                                              100,
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
            color=color2, label='$\\omega_2$')
ax4[1].plot(t_arr_ref, state_hist_ref[5, :]*180/np.pi,
            color=color3, label='$\\omega_3$')
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
np.random.seed(1998)
for n_draw in range(n_draws):
    print(n_draw)
    # Start at rest
    initial_rates = np.zeros((3,))
    initial_rates[1] = -n/np.pi*180
    initial_rates += np.random.normal(0, scale=1, size=3)
    # Draw random initial attitude in each direction
    initial_att = np.random.normal(0, scale=10, size=3)
    t_arr, state_hist = controlled_attitude.propagate_EoM(
        0, .2*P, 100, initial_rates, initial_att)
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
ax4[0].axhline(-.1, color='grey', alpha=1)
ax4[1].axhline(-n/np.pi*180, color='grey', alpha=1, label='-n')
ax4[0].legend()
ax4[1].legend()

# %% Simulate controller behaviour with imperfect sensors
# Initialise an imperfect estimator that straightforwardly takes the
# measurement at each epoch and predicts the control input that way
perfect_estimator = SimpleEstimator(controller)
controlled_attitude = SpacecraftAttitude(estimator=perfect_estimator,
                                         Td_std=np.zeros((3,)),
                                         Td_mean=1e-3*np.ones((3,)))
# Reference scenario
initial_rates = np.zeros((3,))
initial_rates[1] = -n/np.pi*180
initial_att = np.array([10, 10, 10])
t_arr_ref, state_hist_ref = controlled_attitude.propagate_EoM(0,
                                                              .2*P,
                                                              60,
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
# Plot true and estimated states
# True states
ax3[0].plot(t_arr_ref, state_hist_ref[0, :]*180/np.pi,
            color=color1, label='$\\theta_1$')
ax3[0].plot(t_arr_ref, state_hist_ref[1, :]*180/np.pi,
            color=color2, label='$\\theta_2$')
ax3[0].plot(t_arr_ref, state_hist_ref[2, :]*180/np.pi,
            color=color3, label='$\\theta_3$')
# Estimated states
state_est = np.array(controlled_attitude.estimator.predict_state_hist_cont)
t_est = controlled_attitude.estimator.predict_t_cont
ax3[0].plot(t_est, state_est[0, :]*180/np.pi,
            color=color1, label='Est. $\\theta_1$', linestyle='dashed')
ax3[0].plot(t_est, state_est[1, :]*180/np.pi,
            color=color2, label='Est. $\\theta_2$', linestyle='dashed')
ax3[0].plot(t_est, state_est[2, :]*180/np.pi,
            color=color3, label='Est. $\\theta_3$', linestyle='dashed')

ax3[0].set_xlabel('Time [s]')
ax3[0].set_ylabel('Euler angle [$^{\\circ}$]')
ax3[0].set_yscale('symlog')

# True states
ax3[1].plot(t_arr_ref, state_hist_ref[3, :]*180/np.pi,
            color=color1, label='$\\omega_1$')
ax3[1].plot(t_arr_ref, state_hist_ref[4, :]*180/np.pi,
            color=color2, label='$\\omega_2$')
ax3[1].plot(t_arr_ref, state_hist_ref[5, :]*180/np.pi,
            color=color3, label='$\\omega_3$')
# Estimated states
ax3[1].plot(t_est, state_est[3, :]*180/np.pi,
            color=color1, label='Est. $\\omega_1$', linestyle='dashed')
ax3[1].plot(t_est, state_est[4, :]*180/np.pi,
            color=color2, label='Est. $\\omega_2$', linestyle='dashed')
ax3[1].plot(t_est, state_est[5, :]*180/np.pi,
            color=color3, label='Est. $\\omega_3$', linestyle='dashed')
ax3[1].set_xlabel('Time [s]')
ax3[1].set_ylabel('Rotational velocity [$^{\\circ}/s$]')
ax3[0].axhline(.1, color='grey', alpha=1, label='Accuracy\nrequirement')
ax3[0].axhline(-.1, color='grey', alpha=1)
ax3[1].axhline(-n/np.pi*180, color='grey', alpha=1, label='-n')
ax3[0].legend()
ax3[1].legend()

# Now with no mean torque, but random torques
# Initialise an imperfect estimator that straightforwardly takes the
# measurement at each epoch and predicts the control input that way
perfect_estimator = SimpleEstimator(controller)
controlled_attitude = SpacecraftAttitude(estimator=perfect_estimator)
# Reference scenario
initial_rates = np.zeros((3,))
initial_rates[1] = -n/np.pi*180
initial_att = np.array([10, 10, 10])
t_arr_ref, state_hist_ref = controlled_attitude.propagate_EoM(0,
                                                              .2*P,
                                                              1,
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
# Plot true and estimated states
# True states
ax3[0].plot(t_arr_ref, state_hist_ref[0, :]*180/np.pi,
            color=color1, label='$\\theta_1$')
ax3[0].plot(t_arr_ref, state_hist_ref[1, :]*180/np.pi,
            color=color2, label='$\\theta_2$')
ax3[0].plot(t_arr_ref, state_hist_ref[2, :]*180/np.pi,
            color=color3, label='$\\theta_3$')
# Estimated states
state_est = np.array(controlled_attitude.estimator.predict_state_hist_cont)
t_est = controlled_attitude.estimator.predict_t_cont
ax3[0].plot(t_est, state_est[0, :]*180/np.pi,
            color=color1, label='Est. $\\theta_1$', linestyle='dashed')
ax3[0].plot(t_est, state_est[1, :]*180/np.pi,
            color=color2, label='Est. $\\theta_2$', linestyle='dashed')
ax3[0].plot(t_est, state_est[2, :]*180/np.pi,
            color=color3, label='Est. $\\theta_3$', linestyle='dashed')

ax3[0].set_xlabel('Time [s]')
ax3[0].set_ylabel('Euler angle [$^{\\circ}$]')
ax3[0].set_yscale('symlog')

# True states
ax3[1].plot(t_arr_ref, state_hist_ref[3, :]*180/np.pi,
            color=color1, label='$\\omega_1$')
ax3[1].plot(t_arr_ref, state_hist_ref[4, :]*180/np.pi,
            color=color2, label='$\\omega_2$')
ax3[1].plot(t_arr_ref, state_hist_ref[5, :]*180/np.pi,
            color=color3, label='$\\omega_3$')
# Estimated states
ax3[1].plot(t_est, state_est[3, :]*180/np.pi,
            color=color1, label='Est. $\\omega_1$', linestyle='dashed')
ax3[1].plot(t_est, state_est[4, :]*180/np.pi,
            color=color2, label='Est. $\\omega_2$', linestyle='dashed')
ax3[1].plot(t_est, state_est[5, :]*180/np.pi,
            color=color3, label='Est. $\\omega_3$', linestyle='dashed')
ax3[1].set_xlabel('Time [s]')
ax3[1].set_ylabel('Rotational velocity [$^{\\circ}/s$]')
ax3[0].axhline(.1, color='grey', alpha=1, label='Accuracy\nrequirement')
ax3[0].axhline(-.1, color='grey', alpha=1)
ax3[1].axhline(-n/np.pi*180, color='grey', alpha=1, label='-n')
ax3[0].legend()
ax3[1].legend()


# %% Now do the same scenario, but with the Kalman filter
# Initialise the Kalman filter
kalman_estimator = KalmanEstimator(controller)
sensor = Sensor()
controlled_attitude = SpacecraftAttitude(estimator=kalman_estimator,
                                         sensor=sensor,
                                         Td_std=np.zeros((3,)),
                                         Td_mean=1e-3*np.ones((3,)))
# Reference scenario
initial_rates = np.zeros((3,))
initial_rates[1] = -n/np.pi*180
initial_att = np.array([10, 10, 10])


def dt_over_time1(t):
    return .1


t_arr_ref, state_hist_ref = controlled_attitude.propagate_EoM(0,
                                                              # .2*P,
                                                              100,
                                                              dt_over_time1,
                                                              initial_rates,
                                                              initial_att)
t_sett_ref1 = settling_time(t_arr_ref, state_hist_ref[0, :],
                            0, .1/180*np.pi)
t_sett_ref2 = settling_time(t_arr_ref, state_hist_ref[1, :],
                            0, .1/180*np.pi)
t_sett_ref3 = settling_time(t_arr_ref, state_hist_ref[2, :],
                            0, .1/180*np.pi)
# %% Plot the result
color1 = 'tab:blue'
color2 = 'tab:orange'
color3 = 'tab:green'
fig3, ax3 = plt.subplots(1, 2)
# Plot true, estimated states and measurements
measurements = controlled_attitude.sensor.measurements
t_meas = controlled_attitude.sensor.t_arr

# True states
ax3[0].plot(t_arr_ref, state_hist_ref[0, :]*180/np.pi,
            color=color1, label='$\\theta_1$')
ax3[0].plot(t_arr_ref, state_hist_ref[1, :]*180/np.pi,
            color=color2, label='$\\theta_2$')
ax3[0].plot(t_arr_ref, state_hist_ref[2, :]*180/np.pi,
            color=color3, label='$\\theta_3$')
# Estimated states
state_est = np.array(controlled_attitude.estimator.predict_state_hist_cont)
t_est = controlled_attitude.estimator.predict_t_cont
ax3[0].plot(t_est, state_est[0, :]*180/np.pi,
            color=color1, label='Est. $\\theta_1$', linestyle='dashed')
ax3[0].plot(t_est, state_est[1, :]*180/np.pi,
            color=color2, label='Est. $\\theta_2$', linestyle='dashed')
ax3[0].plot(t_est, state_est[2, :]*180/np.pi,
            color=color3, label='Est. $\\theta_3$', linestyle='dashed')
# Measurements
# ax3[0].scatter(t_meas, measurements[0, :]*180/np.pi,
#                color=color1, label='Meas. $\\theta_1$',
#                marker='x', alpha=.1)
# ax3[0].scatter(t_meas, measurements[1, :]*180/np.pi,
#                color=color2, label='Meas. $\\theta_2$',
#                marker='x', alpha=.1)
# ax3[0].scatter(t_meas, measurements[2, :]*180/np.pi,
#                color=color3, label='Meas. $\\theta_3$',
#                marker='x', alpha=.1)

ax3[0].set_xlabel('Time [s]')
ax3[0].set_ylabel('Euler angle [$^{\\circ}$]')
ax3[0].set_yscale('symlog')

# True states
ax3[1].plot(t_arr_ref, state_hist_ref[3, :]*180/np.pi,
            color=color1, label='$\\omega_1$')
ax3[1].plot(t_arr_ref, state_hist_ref[4, :]*180/np.pi,
            color=color2, label='$\\omega_2$')
ax3[1].plot(t_arr_ref, state_hist_ref[5, :]*180/np.pi,
            color=color3, label='$\\omega_3$')
# Estimated states
ax3[1].plot(t_est, state_est[3, :]*180/np.pi,
            color=color1, label='Est. $\\omega_1$', linestyle='dashed')
ax3[1].plot(t_est, state_est[4, :]*180/np.pi,
            color=color2, label='Est. $\\omega_2$', linestyle='dashed')
ax3[1].plot(t_est, state_est[5, :]*180/np.pi,
            color=color3, label='Est. $\\omega_3$', linestyle='dashed')
# Measurements
# ax3[1].scatter(t_meas, measurements[3, :]*180/np.pi,
#                color=color1, label='Meas. $\\omega_1$',
#                marker='x', alpha=.1)
# ax3[1].scatter(t_meas, measurements[4, :]*180/np.pi,
#                color=color2, label='Meas. $\\omega_2$',
#                marker='x', alpha=.1)
# ax3[1].scatter(t_meas, measurements[5, :]*180/np.pi,
#                color=color3, label='Meas. $\\omega_3$',
#                marker='x', alpha=.1)


ax3[1].set_xlabel('Time [s]')
ax3[1].set_ylabel('Rotational velocity [$^{\\circ}/s$]')
ax3[0].axhline(.1, color='grey', alpha=1, label='Accuracy\nrequirement')
ax3[0].axhline(-.1, color='grey', alpha=1)
ax3[1].axhline(-n/np.pi*180, color='grey', alpha=1, label='-n')
ax3[0].legend()
ax3[1].legend()

# Plot also the estimated/predicted state error; do this by interpolating the
# true state onto the timestamps for the estimated state and taking their
# difference
true_state_interp = sp.interpolate.CubicSpline(
    t_arr_ref,
    state_hist_ref,
    axis=1)(t_est)
err = state_est - true_state_interp
# Also extract the estimated errors
tk_hist = controlled_attitude.estimator.t_hist
err_est = np.zeros((6, tk_hist.shape[0]))
# Extract the variances
for i in range(6):
    err_est[i, :] = np.sqrt(controlled_attitude.estimator.P_kk_hist[i, i, :])
fig4, ax4 = plt.subplots(1, 2)
ax4[0].plot(t_est, err[0, :]*180/np.pi,
            color=color1, label='Error in $\\theta_1$')
ax4[0].plot(t_est, err[1, :]*180/np.pi,
            color=color2, label='Error in $\\theta_2$')
ax4[0].plot(t_est, err[2, :]*180/np.pi,
            color=color3, label='Error in $\\theta_3$')

ax4[0].plot(tk_hist, err_est[0, :]*180/np.pi,
            color=color1, label='Est. error in $\\theta_1$',
            linestyle='dashed')
ax4[0].plot(tk_hist, err_est[1, :]*180/np.pi,
            color=color2, label='Est. error in $\\theta_2$',
            linestyle='dashed')
ax4[0].plot(tk_hist, err_est[2, :]*180/np.pi,
            color=color3, label='Est. error in $\\theta_3$',
            linestyle='dashed')
ax4[0].plot(tk_hist, -err_est[0, :]*180/np.pi,
            color=color1, linestyle='dashed')
ax4[0].plot(tk_hist, -err_est[1, :]*180/np.pi,
            color=color2, linestyle='dashed')
ax4[0].plot(tk_hist, -err_est[2, :]*180/np.pi,
            color=color3, linestyle='dashed')

ax4[0].set_xlabel('Time [s]')
ax4[0].set_ylabel('Error [$^{\circ}$]')
ax4[0].set_ylim((-0.01, 0.01))
ax4[0].set_yscale('symlog')
ax4[0].legend()

ax4[1].plot(t_est, err[3, :]*180/np.pi,
            color=color1, label='Error in $\\omega_1$')
ax4[1].plot(t_est, err[4, :]*180/np.pi,
            color=color2, label='Error in $\\omega_2$')
ax4[1].plot(t_est, err[5, :]*180/np.pi,
            color=color3, label='Error in $\\omega_3$')

ax4[1].plot(tk_hist, err_est[3, :]*180/np.pi,
            color=color1, label='Est. error in $\\omega_1$',
            linestyle='dashed')
ax4[1].plot(tk_hist, err_est[4, :]*180/np.pi,
            color=color2, label='Est. error in $\\omega_2$',
            linestyle='dashed')
ax4[1].plot(tk_hist, err_est[5, :]*180/np.pi,
            color=color3, label='Est. error in $\\omega_3$',
            linestyle='dashed')
ax4[1].plot(tk_hist, -err_est[3, :]*180/np.pi,
            color=color1, linestyle='dashed')
ax4[1].plot(tk_hist, -err_est[4, :]*180/np.pi,
            color=color2, linestyle='dashed')
ax4[1].plot(tk_hist, -err_est[5, :]*180/np.pi,
            color=color3, linestyle='dashed')

ax4[1].set_xlabel('Time [s]')
ax4[1].set_ylabel('Error [$^{\circ}$/s]')
ax4[1].set_ylim((-0.0001, 0.0001))
ax4[1].set_yscale('symlog')
ax4[1].legend()


# %% Now do noisy but unbiased disturbance torques
# Initialise the Kalman filter
kalman_estimator = KalmanEstimator(controller)
sensor = Sensor()
controlled_attitude = SpacecraftAttitude(estimator=kalman_estimator,
                                         sensor=sensor,
                                         Td_std=1e-3*np.ones((3,)))


def dt_over_time2(t):
    if t < 180:
        return .1
    elif t < 3000:
        return 10


# Reference scenario
initial_rates = np.zeros((3,))
initial_rates[1] = -n/np.pi*180
initial_att = np.array([10, 10, 10])
t_arr_ref, state_hist_ref = controlled_attitude.propagate_EoM(0,
                                                              10*P,
                                                              # 100,
                                                              dt_over_time2,
                                                              initial_rates,
                                                              initial_att)
t_sett_ref1 = settling_time(t_arr_ref, state_hist_ref[0, :],
                            0, .1/180*np.pi)
t_sett_ref2 = settling_time(t_arr_ref, state_hist_ref[1, :],
                            0, .1/180*np.pi)
t_sett_ref3 = settling_time(t_arr_ref, state_hist_ref[2, :],
                            0, .1/180*np.pi)
# %% Plot the result
color1 = 'tab:blue'
color2 = 'tab:orange'
color3 = 'tab:green'
fig3, ax3 = plt.subplots(1, 2)
# Plot true, estimated states and measurements
measurements = controlled_attitude.sensor.measurements
t_meas = controlled_attitude.sensor.t_arr

# True states
ax3[0].plot(t_arr_ref, state_hist_ref[0, :]*180/np.pi,
            color=color1, label='$\\theta_1$')
ax3[0].plot(t_arr_ref, state_hist_ref[1, :]*180/np.pi,
            color=color2, label='$\\theta_2$')
ax3[0].plot(t_arr_ref, state_hist_ref[2, :]*180/np.pi,
            color=color3, label='$\\theta_3$')
# Estimated states
state_est = np.array(controlled_attitude.estimator.predict_state_hist_cont)
t_est = controlled_attitude.estimator.predict_t_cont
ax3[0].plot(t_est, state_est[0, :]*180/np.pi,
            color=color1, label='Est. $\\theta_1$', linestyle='dashed')
ax3[0].plot(t_est, state_est[1, :]*180/np.pi,
            color=color2, label='Est. $\\theta_2$', linestyle='dashed')
ax3[0].plot(t_est, state_est[2, :]*180/np.pi,
            color=color3, label='Est. $\\theta_3$', linestyle='dashed')
# Measurements
# ax3[0].scatter(t_meas, measurements[0, :]*180/np.pi,
#                color=color1, label='Meas. $\\theta_1$',
#                marker='x', alpha=.2)
# ax3[0].scatter(t_meas, measurements[1, :]*180/np.pi,
#                color=color2, label='Meas. $\\theta_2$',
#                marker='x', alpha=.2)
# ax3[0].scatter(t_meas, measurements[2, :]*180/np.pi,
#                color=color3, label='Meas. $\\theta_3$',
#                marker='x', alpha=.2)

ax3[0].set_xlabel('Time [s]')
ax3[0].set_ylabel('Euler angle [$^{\\circ}$]')
ax3[0].set_yscale('symlog')

# True states
ax3[1].plot(t_arr_ref, state_hist_ref[3, :]*180/np.pi,
            color=color1, label='$\\omega_1$')
ax3[1].plot(t_arr_ref, state_hist_ref[4, :]*180/np.pi,
            color=color2, label='$\\omega_2$')
ax3[1].plot(t_arr_ref, state_hist_ref[5, :]*180/np.pi,
            color=color3, label='$\\omega_3$')
# Estimated states
ax3[1].plot(t_est, state_est[3, :]*180/np.pi,
            color=color1, label='Est. $\\omega_1$', linestyle='dashed')
ax3[1].plot(t_est, state_est[4, :]*180/np.pi,
            color=color2, label='Est. $\\omega_2$', linestyle='dashed')
ax3[1].plot(t_est, state_est[5, :]*180/np.pi,
            color=color3, label='Est. $\\omega_3$', linestyle='dashed')
# Measurements
# ax3[1].scatter(t_meas, measurements[3, :]*180/np.pi,
#                color=color1, label='Meas. $\\omega_1$',
#                marker='x', alpha=.2)
# ax3[1].scatter(t_meas, measurements[4, :]*180/np.pi,
#                color=color2, label='Meas. $\\omega_2$',
#                marker='x', alpha=.2)
# ax3[1].scatter(t_meas, measurements[5, :]*180/np.pi,
#                color=color3, label='Meas. $\\omega_3$',
#                marker='x', alpha=.2)


ax3[1].set_xlabel('Time [s]')
ax3[1].set_ylabel('Rotational velocity [$^{\\circ}/s$]')
ax3[0].axhline(.1, color='grey', alpha=1, label='Accuracy\nrequirement')
ax3[0].axhline(-.1, color='grey', alpha=1)
ax3[1].axhline(-n/np.pi*180, color='grey', alpha=1, label='-n')
ax3[0].legend(loc="upper right")
ax3[1].legend(loc="upper right")

# Plot also the estimated/predicted state error; do this by interpolating the
# true state onto the timestamps for the estimated state and taking their
# difference
true_state_interp = sp.interpolate.CubicSpline(
    t_arr_ref,
    state_hist_ref,
    axis=1)(t_est)
err = state_est - true_state_interp
# Also extract the estimated errors
tk_hist = controlled_attitude.estimator.t_hist
err_est = np.zeros((6, tk_hist.shape[0]))
# Extract the variances
for i in range(6):
    err_est[i, :] = np.sqrt(controlled_attitude.estimator.P_kk_hist[i, i, :])
fig4, ax4 = plt.subplots(1, 2)
ax4[0].plot(t_est, err[0, :]*180/np.pi,
            color=color1, label='Error in $\\theta_1$')
ax4[0].plot(t_est, err[1, :]*180/np.pi,
            color=color2, label='Error in $\\theta_2$')
ax4[0].plot(t_est, err[2, :]*180/np.pi,
            color=color3, label='Error in $\\theta_3$')

ax4[0].plot(tk_hist, err_est[0, :]*180/np.pi,
            color=color1, label='Est. error in $\\theta_1$',
            linestyle='dashed')
ax4[0].plot(tk_hist, err_est[1, :]*180/np.pi,
            color=color2, label='Est. error in $\\theta_2$',
            linestyle='dashed')
ax4[0].plot(tk_hist, err_est[2, :]*180/np.pi,
            color=color3, label='Est. error in $\\theta_3$',
            linestyle='dashed')
ax4[0].plot(tk_hist, -err_est[0, :]*180/np.pi,
            color=color1, linestyle='dashed')
ax4[0].plot(tk_hist, -err_est[1, :]*180/np.pi,
            color=color2, linestyle='dashed')
ax4[0].plot(tk_hist, -err_est[2, :]*180/np.pi,
            color=color3, linestyle='dashed')

ax4[0].set_xlabel('Time [s]')
ax4[0].set_ylabel('Error [$^{\circ}$]')
ax4[0].set_ylim((-0.1, 0.1))
ax4[0].set_yscale('symlog')
ax4[0].legend()

ax4[1].plot(t_est, err[3, :]*180/np.pi,
            color=color1, label='Error in $\\omega_1$',
            alpha=.5)
ax4[1].plot(t_est, err[4, :]*180/np.pi,
            color=color2, label='Error in $\\omega_2$',
            alpha=.5)
ax4[1].plot(t_est, err[5, :]*180/np.pi,
            color=color3, label='Error in $\\omega_3$',
            alpha=.5)

ax4[1].plot(tk_hist, err_est[3, :]*180/np.pi,
            color=color1, label='Est. error in $\\omega_1$',
            linestyle='dashed')
ax4[1].plot(tk_hist, err_est[4, :]*180/np.pi,
            color=color2, label='Est. error in $\\omega_2$',
            linestyle='dashed')
ax4[1].plot(tk_hist, err_est[5, :]*180/np.pi,
            color=color3, label='Est. error in $\\omega_3$',
            linestyle='dashed')
ax4[1].plot(tk_hist, -err_est[3, :]*180/np.pi,
            color=color1, linestyle='dashed')
ax4[1].plot(tk_hist, -err_est[4, :]*180/np.pi,
            color=color2, linestyle='dashed')
ax4[1].plot(tk_hist, -err_est[5, :]*180/np.pi,
            color=color3, linestyle='dashed')

ax4[1].set_xlabel('Time [s]')
ax4[1].set_ylabel('Error [$^{\circ}$/s]')
ax4[1].set_ylim((-1e-4, 1e-4))
ax4[1].set_yscale('symlog')
ax4[1].legend()
