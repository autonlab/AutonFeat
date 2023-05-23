import numpy as np
import sys
package_path = '/Users/dhruvsrikanth/Work/CMU/AutoFeat'
sys.path.append(package_path)

import autofeat as aft


def main():
    # Random data
    n_samples = 100
    x = np.random.rand(n_samples)

    # Create sliding window
    ws = 10
    ss = 10
    window = aft.SlidingWindow(window_size=ws, step_size=ss)

    # Create transform
    tf = aft.MeanTransform()

    # Get featurizer
    featurizer = window.use(tf)

    # Get features
    features = featurizer(x)

    # Print features
    print(window)
    print(tf)
    print(features)


if __name__ == '__main__':
    main()