import sys
sys.path.append("../../../AutonFeat")

import numpy as np
import matplotlib.pyplot as plt
from autonfeat.preprocess.functional import delta_tf


def main():
    # Define signal
    time = np.linspace(0, 10, 1000) # secs
    freq = 500                      # Hz
    dc_offset = 5                   # V
    ac_amp = 2                      # V

    signal = dc_offset + ac_amp * np.sin(2 * np.pi * freq * time)

    # Define half-wave rectifier
    half_wave_rectifier = lambda x_i: np.maximum(x_i, 0)

    delta = dc_offset # Amount to shift by

    # Preprocess signal
    signal_transformed = delta_tf(signal, delta=delta)

    # Apply half-wave rectifier
    system_output = half_wave_rectifier(signal_transformed)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(10, 6))

    # Plot signal and output of half-wave rectifier (before delta transform)
    ax1[0].plot(time, signal, label='Signal')
    ax1[0].set_xlabel('Time (s)')
    ax1[0].set_ylabel('Voltage (V)')
    ax1[0].set_title('Original Signal')
    ax1[0].grid(True)
    ax1[0].legend()

    ax1[1].plot(time, half_wave_rectifier(signal), color='orange', label='Output')
    ax1[1].set_xlabel('Time (s)')
    ax1[1].set_ylabel('Voltage (V)')
    ax1[1].set_title('Output of Half-Wave Rectifier (Before Delta Transform)')
    ax1[1].grid(True)
    ax1[1].legend()

    # Plot signal and output of half-wave rectifier (after delta transform)
    ax2[0].plot(time, signal_transformed, label='Signal')
    ax2[0].set_xlabel('Time (s)')
    ax2[0].set_ylabel('Voltage (V)')
    ax2[0].set_title('Signal After Delta Transform')
    ax2[0].grid(True)
    ax2[0].legend()

    ax2[1].plot(time, system_output, color='orange', label='Output')
    ax2[1].set_xlabel('Time (s)')
    ax2[1].set_ylabel('Voltage (V)')
    ax2[1].set_title('Output of Half-Wave Rectifier (After Delta Transform)')
    ax2[1].grid(True)
    ax2[1].legend()

    plt.tight_layout()

    # Save figure
    fig.savefig('../../docs/assets/delta_f_visualize.png')


if __name__ == '__main__':
    main()
