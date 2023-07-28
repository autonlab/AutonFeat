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
import autonfeat.preprocess.functional as PF


def main():
    # Parameters for the normal distribution
    mu = 10     # Mean
    sigma = 1   # Standard deviation
    n_samples = 10000

    # Generate random samples from the normal distribution
    samples = np.random.normal(mu, sigma, n_samples)

    # Preprocess signal
    transformed_samples = PF.delta_mean_tf(samples)

    # Compute the range and pdf for plotting
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, n_samples)
    pdf = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2))

    # Compute the expected range and pdf for plotting
    x_shifted = x - mu
    transformed_pdf = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x_shifted)**2 / (2 * sigma**2))

    # Plot one below the other
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    ax1.hist(samples, bins=50, density=True, alpha=0.7, color='grey')
    ax1.plot(x, pdf, color='blue', linewidth=2)
    ax1.axvline(x=mu, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('x')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Normal Distribution Centered at {:.2f}'.format(mu))

    ax2.hist(transformed_samples, bins=50, density=True, alpha=0.7, color='grey')
    ax2.plot(x_shifted, transformed_pdf, color='blue', linewidth=2)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Normal Distribution Centered at {:.2f}'.format(0))

    plt.tight_layout()

    # Save figure
    fig.savefig('../../docs/assets/delta_mean_f_visualize.png')


if __name__ == '__main__':
    main()
