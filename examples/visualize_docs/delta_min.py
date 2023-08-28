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
