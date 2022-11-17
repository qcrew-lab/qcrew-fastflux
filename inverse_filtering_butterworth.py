import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from bias_tee_opx_transfer import get_interpolated_transfer_functions

# Now that we have the transfer function from the OPX + bias tee and we know how
# to apply transfer functions, I want to solve the inverse filtering problem.

# Get function for HP butterworth filter
f_cutoff = 9e3 # cutoff frequency in Hz
w_cutoff = 2*np.pi*f_cutoff
N = 1
b, a = sig.butter(N, w_cutoff, btype='highpass', analog=True, output='ba')
#w, h = sig.freqs(b, a)
transfer_func = lambda f: sig.freqresp((b,a), w=2*np.pi*f)[1]
# define output pulse
dt = 1e-9  # time step
def get_input_pulse():
    # Get array of ampx values of the output pulse
    V = 0.01  # pulse ampx as defined in the modes.yaml
    on_pulse_len = int(100)
    padding_left = int(100)
    padding_right = int(100)
    total_len = padding_left + on_pulse_len + padding_right
    pulse = []
    for i in range(total_len):
        if i < padding_left or i > padding_left + on_pulse_len:
            pulse.append(0)
        else:
            #pulse.append(-V if (i//100)%2 else V)
            pulse.append(V)
    return pulse

# Now apply reverse filtering
in_pulse = get_input_pulse()
in_pulse_fft = np.fft.fft(in_pulse)
fft_freqs = np.fft.fftfreq(len(in_pulse), d=dt)
print(list(fft_freqs))
# apply inverse transfer function
out_pulse_fft = [in_pulse_fft[i]*transfer_func(abs(fft_freqs)[i]) for i in range(len(fft_freqs))]
out_pulse = np.fft.ifft(out_pulse_fft)

plt.plot([20*np.log10(np.abs(transfer_func(fft_freqs[i]))) for i in range(len(fft_freqs))])
plt.ylabel(r"$|G|_{dB}$")
plt.xlabel("frequency (MHz)")
plt.legend()
plt.grid()
plt.show()

plt.plot([np.angle(transfer_func(fft_freqs[i])) for i in range(len(fft_freqs))])
plt.ylabel(r"phase of G")
plt.xlabel("frequency (MHz)")
plt.legend()
plt.show()
#plt.plot(in_pulse)
#plt.plot(np.abs(out_pulse))
#plt.show()
#plt.plot(fft_freqs, out_pulse_fft)
#plt.plot(fft_freqs, in_pulse_fft)
#plt.show()