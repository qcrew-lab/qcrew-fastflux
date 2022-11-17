import scipy.signal as sig
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt

dt = 1e-9
fig, axes = plt.subplots(3,1)
# Load response to predistorted pulse
step_output = np.load("bias_tee_step_response/step_response_30us_200kHz_biastee_50krep.npz")["arr_0"]
predistorted_input = np.load("bias_tee_predistorted_step/square_6us_200kHz_biastee.npz")["oct_pulse"]
predistorted_output = np.load("bias_tee_step_response/predistorted_square_6us_200kHz_biastee_10krep.npz")["arr_0"]
axes[0].plot(dt*np.array(range(len(step_output))), step_output)
axes[0].set_title("200kHz bias tee response to step")
axes[1].plot(dt*np.array(range(len(predistorted_input))), predistorted_input)
axes[1].set_title("Predistorted square pulse")
axes[2].plot(dt*np.array(range(len(predistorted_output))), np.array(predistorted_output)/2**12)
axes[2].set_title("200kHz bias tee response to predistorted square pulse")
plt.show()

fig, axes = plt.subplots(3,1)
# Load response to predistorted pulse
step_output = np.load("bias_tee_step_response/step_response_30us_9kHz_biastee_50krep.npz")["arr_0"]
predistorted_input = np.load("bias_tee_predistorted_step/square_30us_9kHz_biastee.npz")["oct_pulse"]
predistorted_output = np.load("bias_tee_step_response/predistorted_square_30us_9kHz_biastee_10krep.npz")["arr_0"]
axes[0].plot(dt*np.array(range(len(step_output))), step_output)
axes[0].set_title("9kHz bias tee response to step")
axes[1].plot(dt*np.array(range(len(predistorted_input))), predistorted_input)
axes[1].set_title("Predistorted square pulse")
axes[2].plot(dt*np.array(range(len(predistorted_output))), np.array(predistorted_output)/2**12)
axes[2].set_title("9kHz bias tee response to predistorted square pulse")
plt.show()