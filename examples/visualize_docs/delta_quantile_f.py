# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.


import sys
sys.path.append("../../../AutonFeat")

import numpy as np
import matplotlib.pyplot as plt
import autonfeat.functional as F
import autonfeat.preprocess.functional as PF


def main():
    # Generate data
    n_samples = 1000
    x = np.random.normal(-5, 5, n_samples)

    # Preprocess data
    x_shifted = PF.delta_quantile_tf(x, q=0.25)

    # Plot normal and shifted data
    original_quantile = F.quantile_tf(x, 0.25)
    shifted_quantile = F.quantile_tf(x_shifted, 0.25)

    plt.figure(figsize=(8, 6))

    plt.plot(x, '.', color='blue', label='Origianl Data')
    plt.axhline(original_quantile, color='red', linestyle='--', linewidth=3, label=f'Original Data 25th quantile = {original_quantile:.2f}')

    plt.plot(x_shifted, '.', color='orange', label='Shifted Data')
    plt.axhline(shifted_quantile, color='green', linestyle='--', linewidth=3, label=f'Shifted Data 25th quantile = {shifted_quantile:.2f}')

    plt.legend()
    plt.title('Delta Quantile Preprocessing')

    plt.tight_layout()

    # Save figure
    plt.savefig('../../docs/assets/delta_quantile_f_visualize.png')


if __name__ == '__main__':
    main()
