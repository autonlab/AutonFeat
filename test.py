import numpy as np
import autofeat as aft


def main():
    # Random data
    n_samples = 5
    x1 = np.random.rand(n_samples)
    x2 = np.random.rand(n_samples)

    # Get features
    y_hat = aft.functional.entropy_tf(x1, x2, base=2)
    print(y_hat)


if __name__ == '__main__':
    main()
