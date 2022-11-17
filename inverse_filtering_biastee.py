import scipy.signal as sig
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt

V = 1
dt = 1e-9  # time step

def get_step_pulse():
    # Get array of ampx values of the output pulse
    total_len = 15000
    on_pulse_len = 6000
    padding_left = 1000
    pulse = []
    for i in range(total_len):
        if i < padding_left or i > padding_left + on_pulse_len:
            pulse.append(0)
        else:
            #pulse.append(-V if (i//2200)%2 else V)
            pulse.append(V)
    return np.array([dt*i for i in range(total_len)]), pulse

predistortion_iteration = 1
folder_name = "biastee_9kHz_20k_reps_6us_length/"
intv = np.array([range(1500, 6900)]) # fit interval
fit_guess = (0.0511, 0.06, 3e-6)

y_file = folder_name + "y%d.npz" % predistortion_iteration
y = np.array(np.load(y_file)["arr_0"]/2**12)
ts = dt*np.array(range(len(y)))

# Obtain first correction for the input pulse
#offset = np.average(y[0:20].flatten()) # voltage offset of the pulse
#y -= offset # correct pulse for voltage offset (?)

# Fit function
def exp_func(t, A, B, tau):
    return A + B*np.exp(-t/tau)
#print(ts[interval])
popt = opt.curve_fit(exp_func, ts[intv].flatten(), 
                    y[intv].flatten(), p0=fit_guess)[0]

A, B, tau = tuple(popt)

# Extracting pole behaviour from the fit
lamb = 2*A*tau + 2*B*tau + A*dt
a1 = (A*tau + 2*B*tau - A*dt)/lamb
b0 = (2*tau + dt)/lamb
b1 = (-2*tau + dt)/lamb
H_correction =  sig.dlti([b0, b1], [1, -a1], dt=dt)

# Get predistorted pulse
if predistortion_iteration == 0:
    ts, target_pulse = get_step_pulse()
else:
    target_pulse = -np.array(np.load(folder_name + "x%d" % predistortion_iteration + ".npz")["oct_pulse"]/2**12)
    ts = dt*np.array(range(len(target_pulse)))
_, predist_target_pulse = sig.dlsim(H_correction, target_pulse, t = ts)

fig, axes = plt.subplots(3, 1)
axes[0].set_title("Filter response")
axes[0].plot(dt*np.array(range(len(y))), y, label = "output")
axes[0].legend()
axes[1].set_title("Exponential fit to extract poles")
axes[1].plot(ts[intv].flatten(), y[intv].flatten())
axes[1].plot(ts[intv].flatten(), exp_func(ts[intv].flatten(), *list(popt)))
axes[2].set_title('Predistorted pulse')
axes[2].plot(ts, target_pulse)
axes[2].plot(ts, predist_target_pulse)
plt.show()
# Normalize pulse before saving
predist_target_pulse /= np.max(np.abs(predist_target_pulse))
np.savez(folder_name + "x%d" % (predistortion_iteration + 1), 
         oct_pulse = predist_target_pulse.flatten())

print("##########  REPORT  ##########")
print("Fit interval: ", np.min(intv), "to", np.max(intv))
print("Fit initial guess: ", fit_guess)
print("Fit optimized parameters: ", popt)

# Load response to predistorted pulse
# predistorted_output = np.load("bias_tee_step_response/predistorted_square_6us_200kHz_biastee.npz")["arr_0"]
# axes[2].plot(dt*np.array(range(len(predistorted_output))), -10*np.array(predistorted_output)/2**12)
# plt.show()