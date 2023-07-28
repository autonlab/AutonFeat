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
    # Generate data
    n_samples = 1000
    x = np.random.normal(-5, 5, n_samples)

    # Create a preprocessor
    preprocessor = aft.preprocess.DeltaQuantilePreprocessor()

    # Preprocess data
    x_shifted = preprocessor(x, q=0.25)

    # Plot normal and shifted data
    original_quantile = aft.functional.quantile_tf(x, 0.25)
    shifted_quantile = aft.functional.quantile_tf(x_shifted, 0.25)

    plt.figure(figsize=(8, 6))

    plt.plot(x, '.', color='blue', label='Origianl Data')
    plt.axhline(original_quantile, color='red', linestyle='--', linewidth=3, label=f'Original Data 25th quantile = {original_quantile:.2f}')

    plt.plot(x_shifted, '.', color='orange', label='Shifted Data')
    plt.axhline(shifted_quantile, color='green', linestyle='--', linewidth=3, label=f'Shifted Data 25th quantile = {shifted_quantile:.2f}')

    plt.legend()
    plt.title('Delta Quantile Preprocessing Transform')

    plt.tight_layout()

    # Save figure
    plt.savefig('../../docs/assets/delta_quantile_visualize.png')


if __name__ == '__main__':
    main()
