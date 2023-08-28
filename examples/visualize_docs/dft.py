# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.


import sys
sys.path.append("../../../AutonFeat")

import numpy as np
import matplotlib.pyplot as plt
import autonfeat as aft


def main():
    start_time = 1  # Start time in seconds
    end_time = 10    # End time in seconds
    sampling_rate = 100  # Number of samples per second
    num_samples = int((end_time - start_time) * sampling_rate)

    # Signal = 2 x sin(2 x pi x t) + sin(10 x 2 x pi x t)
    time = np.linspace(start_time, end_time, num_samples)
    signal = 2 * np.sin(2 * np.pi * time) + np.sin(10 * 2 * np.pi * time)

    # Create Preprocessor
    preprocessor = aft.preprocess.DFTPreprocessor()

    # Preprocess and transform signal
    transformed_signal = preprocessor(signal)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    ax1.plot(time, signal)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("f(t)")
    ax2.plot(np.fft.fftfreq(num_samples, 1 / sampling_rate), np.abs(transformed_signal))
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("| FFT(f(x)) |")

    plt.tight_layout()

    # Save figure
    fig.savefig('../../docs/assets/dft_visualize.png')


if __name__ == '__main__':
    main()
