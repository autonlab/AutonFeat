import sys
package_path = '/Users/dhruvsrikanth/Work/CMU/AutoFeat'
sys.path.append(package_path)

import numpy as np
import matplotlib.pyplot as plt
import autofeat.preprocess.functional as PF


def main():
    # Number of samples
    n_samples = 100

    # Generate sample data
    x1 = np.random.normal(0, 1, n_samples)
    x2 = np.random.normal(5, 5, n_samples)

    shifted_x1 = PF.delta_std_tf(x1)
    shifted_x2 = PF.delta_std_tf(x2)

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
    plt.savefig('../../docs/assets/delta_std_f_visualize.png')


if __name__ == '__main__':
    main()
