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

    # Define signal
    num_samples = 10
    signal = np.arange(1, num_samples + 1)

    lag = 2

    # Preprocess and transform signal
    transformed_signal = PF.lag_tf(signal, lag=lag)
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
    fig.savefig('../../docs/assets/lag_f_visualize.png')


if __name__ == '__main__':
    main()
