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
    # Create a random signal
    time = np.linspace(0, 10, 1000)
    frequency = 500  # Frequency of the signal in Hz
    signal = np.sin(np.exp(np.sin(2 * np.pi * frequency * time)))

    # Create Preprocessor
    preprocessor = aft.preprocess.DeltaMinPreprocessor()

    # Shift the signal by the minimum value
    shifted_signal = preprocessor(signal)

    # Plot the original signal and the shifted signal
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time, signal, label='Original Signal')
    ax.plot(time, shifted_signal, label='Shifted Signal')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Signal')
    ax.legend()
    plt.tight_layout()

    # Save figure
    fig.savefig('../../docs/assets/delta_min_visualize.png')


if __name__ == '__main__':
    main()
