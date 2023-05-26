import numpy as np
import sys
package_path = '/Users/dhruvsrikanth/Work/CMU/AutoFeat'
sys.path.append(package_path)

import autofeat as aft


def main():
    # Random distribution
    n_samples = 100
    dist = np.random.rand(n_samples)

    # Preprocessor
    preprocessor = aft.preprocess.DeltaMaxPreprocessor()

    # Transform distribution
    transformed_dist = preprocessor(dist)

    # Print features
    print(preprocessor)
    print(dist)
    print(transformed_dist)


if __name__ == '__main__':
    main()
