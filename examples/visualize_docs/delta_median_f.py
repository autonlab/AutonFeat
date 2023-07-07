import sys
sys.path.append("../../../AutonFeat")

import numpy as np
import matplotlib.pyplot as plt
import autonfeat.preprocess.functional as PF


def main():
    # Generating example signals for each instrument
    num_samples = 100
    time = np.linspace(0, 1, num_samples)
    guitar_signal = np.sin(2 * np.pi * 10 * time)  # Guitar signal (higher frequency sine wave)
    piano_signal = np.cos(2 * np.pi * 2 * time)  # Piano signal (cosine wave)
    drums_signal = np.random.normal(2, 0.5, num_samples)  # Drums signal (random noise with higher mean)

    # Applying the transform to each signal
    guitar_eq = PF.delta_median_tf(guitar_signal)
    piano_eq = PF.delta_median_tf(piano_signal)
    drums_eq = PF.delta_median_tf(drums_signal)

    # Set up custom line styles
    line_styles = ['-', '--', '-.']

    # Set up custom color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Plotting all original signals on one subplot
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 1, 1)
    for i, signal in enumerate([guitar_signal, piano_signal, drums_signal]):
        plt.plot(time, signal, color=colors[i], linestyle=line_styles[i])

    plt.title('Original Signals')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend(['Guitar', 'Piano', 'Drums'])
    plt.grid(True)

    # Plotting all shifted signals on another subplot
    plt.subplot(2, 1, 2)
    for i, signal in enumerate([guitar_eq, piano_eq, drums_eq]):
        plt.plot(time, signal, color=colors[i], linestyle=line_styles[i])

    plt.title('Shifted Signals')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend(['Guitar', 'Piano', 'Drums'])
    plt.grid(True)

    plt.tight_layout()

    # Save figure
    plt.savefig('../../docs/assets/delta_median_f_visualize.png')


if __name__ == '__main__':
    main()
