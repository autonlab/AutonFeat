# MIT License

# Copyright (c) 2023 Carnegie Mellon University, Auton Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import sys
sys.path.append("../../../AutonFeat")

import numpy as np
import matplotlib.pyplot as plt
import autonfeat as aft


def main():
    # Define signal
    time = np.linspace(0, 10, 1000) # secs
    freq = 500                      # Hz
    dc_offset = 5                   # V
    ac_amp = 2                      # V

    signal = dc_offset + ac_amp * np.sin(2 * np.pi * freq * time)

    # Define half-wave rectifier
    half_wave_rectifier = lambda x_i: np.maximum(x_i, 0)

    # Define delta transform preprocessor
    preprocessor = aft.preprocess.DeltaPreprocessor()

    delta = dc_offset  # Amount to shift by

    # Preprocess signal
    signal_transformed = preprocessor(signal, delta=delta)

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
    fig.savefig('../../docs/assets/delta_visualize.png')


if __name__ == '__main__':
    main()
