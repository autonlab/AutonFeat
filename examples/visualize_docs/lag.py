import sys
package_path = '/Users/dhruvsrikanth/Work/CMU/AutoFeat'
sys.path.append(package_path)

import numpy as np
import matplotlib.pyplot as plt
import autofeat as aft


def main():

    # Define signal
    num_samples = 10
    signal = np.arange(1, num_samples + 1)

    lag = 2

    # Create Preprocessor
    preprocessor = aft.preprocess.LagPreprocessor()

    # Preprocess and transform signal
    transformed_signal = preprocessor(signal, lag=lag)
    transformed_signal = np.nan_to_num(transformed_signal)

    # Plot results
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(signal, label='Original Signal')
    ax.plot(transformed_signal, label='Lag Transformed Signal')
    ax.set_xlabel('Time')
    ax.set_ylabel('Signal')
    ax.set_title('Lag Preprocessor')
    ax.legend()
    ax.grid()

    plt.tight_layout()

    # Save figure
    fig.savefig('../../docs/assets/lag_visualize.png')


if __name__ == '__main__':
    main()
