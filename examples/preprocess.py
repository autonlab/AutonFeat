import autofeat as aft
import numpy as np


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
