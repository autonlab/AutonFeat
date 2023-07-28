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
from autonfeat.preprocess.functional import dft_tf


def main():
    start_time = 1  # Start time in seconds
    end_time = 10    # End time in seconds
    sampling_rate = 100  # Number of samples per second
    num_samples = int((end_time - start_time) * sampling_rate)

    # Signal = 2 x sin(2 x pi x t) + sin(10 x 2 x pi x t)
    time = np.linspace(start_time, end_time, num_samples)
    signal = 2 * np.sin(2 * np.pi * time) + np.sin(10 * 2 * np.pi * time)

    # Preprocess and transform signal
    transformed_signal = dft_tf(signal)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    ax1.plot(time, signal)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("f(t)")

    ax2.plot(
        np.fft.fftfreq(num_samples, 1 / sampling_rate),
        np.abs(transformed_signal)
    )
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("| FFT(f(x)) |")

    plt.tight_layout()

    # Save figure
    fig.savefig('../../docs/assets/dft_f_visualize.png')


if __name__ == '__main__':
    main()
