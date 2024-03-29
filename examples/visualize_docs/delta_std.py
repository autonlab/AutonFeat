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
    # Number of samples
    n_samples = 100

    # Generate sample data
    x1 = np.random.normal(0, 1, n_samples)
    x2 = np.random.normal(5, 5, n_samples)

    # Define preprocessor
    preprocessor = aft.preprocess.DeltaStdPreprocessor()

    shifted_x1 = preprocessor(x1)
    shifted_x2 = preprocessor(x2)

    # Plot original data
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x1, 'b.', label='x1')
    plt.plot(x2, 'r.', label='x2')
    plt.legend()
    plt.title('Original Data')

    # Plot shifted data
    plt.subplot(1, 2, 2)
    plt.plot(shifted_x1, 'b.', label='x1 shifted')
    plt.plot(shifted_x2, 'r.', label='x2 shifted')
    plt.legend()
    plt.title('Shifted Data')

    plt.tight_layout()

    # Save figure
    plt.savefig('../../docs/assets/delta_std_visualize.png')


if __name__ == '__main__':
    main()
