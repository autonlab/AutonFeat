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
